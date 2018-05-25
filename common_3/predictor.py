from model import SSDModel
from converter import Converter
import numpy as np
from PIL import Image

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from metrics import RectWithConf

from rect import Rect
from skimage import color
from keras.applications.vgg16 import preprocess_input
import skimage.transform


class Predictor:
    # def __init__(self, ssd_model, ssd_converter):
    #     self.ssd_model = ssd_model
    #     self.ssd_converter = ssd_converter
    #
    # def predict(self, path, top=10, threshold=None):
    #
    #     res = []
    #
    #     for divisor in self.ssd_model.divisors:
    #
    #         shape = (300 / divisor, 300 / divisor)
    #
    #         img = image.load_img(path, target_size=shape)
    #         img = image.img_to_array(img)
    #         img = img.reshape([1] + list(img.shape))
    #         img = preprocess_input(img)
    #
    #         tensor = self.ssd_model.model.predict(img)
    #         res2 = []
    #         res2 = self.ssd_converter.restore_rects(tensor, self.ssd_model, res2, top=top, threshold=threshold)
    #         res += res2
    #         #print res
    #
    #     #print 42
    #     #print res
    #     confs, rects = [r[0] for r in res], [r[1] for r in res]
    #     print confs
    #     print rects
    #     for rect in rects:
    #         rect.stretch(300, 300)
    #     return sorted([RectWithConf(r, c) for r, c in zip(rects, confs)], key=lambda x: x.conf, reverse=True)
    def __init__(self, ssd_model, pixels_model):
        self.ssd_model = ssd_model
        self.pixels_model = pixels_model

    def _find_components(self, map):
        shape = map.shape
        assert len(shape) == 2

        height, width = shape
        comps = np.zeros(shape, dtype=int)

        def dfs(map, comps, comp, x, y):
            st = [(y, x)]
            comps[y][x] = comp

            while len(st) != 0:
                cur = st[-1]
                st = st[:-1]

                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    cy, cx = cur[0] + dy, cur[1] + dx
                    if 0 <= cx and cx < width and 0 <= cy and cy < height and comps[cy][cx] == 0 and map[cy][cx]:
                        comps[cy][cx] = comp
                        st.append((cy, cx))

        comp = 1
        for y in range(height):
            for x in range(width):
                if map[y][x] and comps[y][x] == 0:
                    dfs(map, comps, comp, x, y)
                    #print comps
                    comp += 1

        print comps
        return comps, comp - 1

    def _find_rects(self, map):
        shape = map.shape
        assert len(shape) == 2
        height, width = shape

        comps, num_comps = self._find_components(map)
        rects = [Rect(width, height, 0, 0) for _ in range(num_comps)]
        for y in range(height):
            for x in range(width):
                if comps[y][x] != 0:
                    rect = comps[y][x] - 1
                    rects[rect].left = min(rects[rect].left, x)
                    rects[rect].right = max(rects[rect].right, x)
                    rects[rect].top = min(rects[rect].top, y)
                    rects[rect].bottom = max(rects[rect].bottom, y)

        #print 'found: ', rects
        return rects

    def _restore_rect(self, rect, tensor_lr, tensor_tb):
        assert tensor_lr.shape == tensor_tb.shape
        assert len(tensor_lr.shape) == 3
        assert isinstance(rect, Rect)
        map_h, map_w, _ = tensor_lr.shape

        sum_l, sum_r = 0., 0.
        for i in range(rect.top, rect.bottom + 1):
            sum_l += tensor_lr[i][rect.left][0]
            sum_r += -tensor_lr[i][rect.right][1]

        sum_t, sum_b = 0., 0.
        for i in range(rect.left, rect.right + 1):
            sum_t += tensor_tb[rect.top][i][0]
            sum_b += - tensor_tb[rect.bottom][i][1]

        height, width = rect.bottom - rect.top + 1, rect.right - rect.left + 1
        ans = Rect(rect.left + sum_l / height, rect.top + sum_t / width,
                   rect.right + 1 + sum_r / height, rect.bottom + 1 + sum_b / width)
        #print ans
        ans.stretch(1.0 / map_h, 1.0 / map_w)
        #print ans
        return ans

    def _get_rects(self, tensor_cls, tensor_lr, tensor_tb):
        assert len(tensor_cls.shape) == 2
        assert len(tensor_lr.shape) == 3
        assert len(tensor_tb.shape) == 3
        map = tensor_cls > 0.5
        #print map
        rects = self._find_rects(map)
        restored_rects = []
        for rect in rects:
            restored_rects.append(self._restore_rect(rect, tensor_lr, tensor_tb))
        #print 'rest', restored_rects
        return restored_rects

    def _softmax(self, cls):
        mx = np.maximum(cls[:,:,:,0], cls[:,:,:,1])
        exp_gr, exp_r = np.exp(cls[:,:,:,0] - mx), np.exp(cls[:,:,:,1] - mx)
        return exp_r / (exp_r + exp_gr)

    def predict(self, img):
        assert len(img.shape) == 4
        _, height, width, depth = img.shape
        img_gray = color.rgb2gray(img[0,:,:,:]).reshape(1, height, width, 1)
        #print np.max(img_gray), np.min(img_gray)
        img_rgb = preprocess_input(np.copy(img))

        pred_cls = self.pixels_model.model.predict(img_gray)
        # print pred_cls[0,:,:,0]
        # print pred_cls[0,:,:,1]
        pred_cls = self._softmax(pred_cls)
        #print pred_cls[0]
        pred_lr, pred_tb = self.ssd_model.model.predict(img_rgb)
        print pred_cls.shape
        print pred_lr.shape, pred_tb.shape
        print pred_lr
        print pred_tb
        if pred_cls.shape != pred_lr.shape:
            pred_cls = skimage.transform.resize(pred_cls[0], pred_lr.shape[1:3]).reshape(pred_lr.shape[:3])
            print 'resized!'

        print np.around(pred_cls[0], decimals=3)

        rects = self._get_rects(pred_cls[0], pred_lr[0], pred_tb[0])
        for rect in rects:
            rect.stretch(height, width)
        return rects
