import numpy as np
import os
from model import SSDModel
from rect import Rect
import random
import skimage.transform

from scipy.misc import imresize

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


class Generator:

    def __init__(self, root, model):
        if not isinstance(model, SSDModel):
            raise Exception('expected SSDModel')
        self.root = root
        self.model = model
        self.max_ratio = model.max_ratio
        self.num_poolings = model.num_poolings

    def _get_possible_windows(self, h, w):
        mn = min(h, w)
        mx = max(h, w)
        mx = min(mx, int(mn * self.max_ratio))
        window = 1 << mn.bit_length()
        res = []
        while window < 2 * mx:
            if window > mn:
                res.append(window)
            window *= 2

        res.append(window)
        return res

    def crop_rect(self, rect):
        assert isinstance(rect, Rect)
        if rect.width() > self.max_ratio * rect.height():
            return rect.center().rect_from_center(rect.height(), rect.height() * self.max_ratio)
        if rect.height() > self.max_ratio * rect.width():
            return rect.center().rect_from_center(rect.width() * self.max_ratio, rect.width())
        return rect.copy()

    def _check_rect(self, rect_in, rect_out):
        if not rect_out.contains(rect_in.center()):
            return False

        if rect_out.area() < rect_in.area():
            return False

        i = rect_in.intersection(rect_out)

        qual_in = i.area() / float(rect_in.area())
        qual_out = i.area() / float(rect_out.area())

        if qual_in > 0.5 and qual_out > 0.5:
            return True

        cropped = self.crop_rect(rect_in)
        cropped.top += 1
        cropped.left += 1

        pts = [cropped.left_top(), cropped.left_bottom(), cropped.right_top(), cropped.right_bottom()]
        cnt = len((filter(lambda x: rect_out.contains(x), pts)))
        return cnt != 1

    def _get_rect_tensor_descs(self, rect, window_rect, cnt):
        assert isinstance(rect, Rect)
        assert isinstance(window_rect, Rect)

        window_size = window_rect.height()
        edge = (window_size >> self.model.num_poolings) / cnt
        center = rect.center()
        sx = int(center.x - window_rect.left)
        sy = int(center.y - window_rect.top)
        x = int((sx >> self.model.num_poolings) / edge)
        y = int((sy >> self.model.num_poolings) / edge)

        res = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                wx = x + dx
                wy = y + dy
                if 0 <= wx and wx + 1 < cnt and 0 <= wy and wy + 1 < cnt:
                    # map coord
                    wrect = Rect(wx * edge, wy * edge, (wx + 2) * edge, (wy + 2) * edge)
                    # window coord
                    wrect.stretch(1 << (self.model.num_poolings), 1 << (self.model.num_poolings))
                    # img coord
                    wrect.move(window_rect.top, window_rect.left)

                    if self._check_rect(rect, wrect):
                        # img coord
                        i = wrect.intersection(rect)
                        qual_out = i.area() / float(wrect.area())
                        if qual_out <= 0:
                            print rect ,wrect, i
                        # window coord
                        i.move(-wrect.top, -wrect.left)
                        # window 0..1 coord
                        i.stretch(1.0 / wrect.height(), 1.0 / wrect.width())

                        res.append((wy, wx, i.left, i.top, i.right, i.bottom, qual_out))

        return res

    def _create_window_for_rect(self, rect, window, img_h, img_w):
        assert isinstance(rect, Rect)
        if window > img_h or window > img_w:
            return None

        assert rect.height() <= img_h and rect.width() <= img_w
        # assert rect.height() <= window and rect.width() <= window

        min_top = max(0, rect.bottom - window)
        max_bottom = min(img_h, rect.top + window)
        max_top = max_bottom - window
        # assert min_top < max_bottom

        min_left = max(0, rect.right - window)
        max_right = min(img_w, rect.left + window)
        max_left = max_right - window
        # assert min_left < max_left

        left = random.randint(min_left, max_left) if min_left < max_left else min_left
        top = random.randint(min_top, max_top) if min_top < max_top else min_top
        return Rect(left, top, left + window, top + window)

    def _process_anno(self, anno, processed_data):
        # anno = {"Rects": [[594,
        # 1081,
        # 326,
        # 575
        #       ],
        #
        #     ],
        #     "Types": [
        #       "UPCE",
        #     ],
        #     "id": "0001",
        #     "name": "0001.jpg",
        #     "path": "Barcodes_1d/UPC-E",
        #     "shape": [
        #       3585,
        #       2661,
        #       3
        #     ]
        #   }

        rects = list(map(lambda r: Rect(r[0], r[2], r[1], r[3]), anno['Rects']))

        img_h, img_w, depth = anno['shape']
        for rect in rects:
            windows = self._get_possible_windows(rect.height(), rect.width())
            for window in windows:
                cnt = 1
                while cnt == 1 or window * cnt <= min(img_h, img_w):

                    window_rect = self._create_window_for_rect(rect, window * cnt, img_h, img_w)
                    if window_rect is None:
                        continue
                    tensor_descs = []
                    for other_rect in rects:
                        tensor_descs += self._get_rect_tensor_descs(other_rect, window_rect, cnt)

                    if not tensor_descs:
                        pass
                        for other_rect in rects:
                            tensor_descs += self._get_rect_tensor_descs(other_rect, window_rect, cnt)

                    if tensor_descs:
                        if cnt not in processed_data:
                            processed_data[cnt] = []

                        desc = {
                            'path': os.path.join(anno['path'], anno['name']),
                            'shape': anno['shape'],
                            'window': window_rect,
                            'tensors': tensor_descs
                        }

                        processed_data[cnt].append(desc)

                    cnt *= 2

    def _get_processed_batches(self, annos, processed_data, batch_size):
        for anno in annos:
            self._process_anno(anno, processed_data)

            if True:
                for key, value in processed_data.items():
                    print key, ':', len(value)

            for key in processed_data.keys():
                val = processed_data[key]
                while len(val) >= batch_size:
                    yield key, val[:batch_size]
                    val = val[batch_size:]

                processed_data[key] = val

                if True:
                    for key2, value2 in processed_data.items():
                        print key2, ':', len(value2)

    def _gen_tensors(self, size, processed_tensors):
        batch_size = len(processed_tensors)

        # input tensor
        inputs = []
        for info in processed_tensors:
            window = info['window']
            img_size = size << self.num_poolings

            img = image.img_to_array(image.load_img(os.path.join(self.root, info['path'])))
            img = img[window.top:window.bottom, window.left:window.right, :]
            z = np.zeros((window.height(), window.width(), 3), dtype=float)
            grid = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
            z[grid] = img[grid]
            img = z

            img = skimage.transform.resize(img, (img_size, img_size), mode='reflect')
            inputs.append(img)
        inputs = preprocess_input(np.array(inputs))

        # output_tensors
        bbox_size = 2 * size - 1
        cls = np.zeros((batch_size, bbox_size, bbox_size, 1))
        qual = np.zeros((batch_size, bbox_size, bbox_size, 1))
        left = np.zeros((batch_size, bbox_size, bbox_size, 1))
        top = np.zeros((batch_size, bbox_size, bbox_size, 1))
        right = np.zeros((batch_size, bbox_size, bbox_size, 1))
        bottom = np.zeros((batch_size, bbox_size, bbox_size, 1))

        for ind, info in enumerate(processed_tensors):
            tensors = info['tensors']
            for tensor in tensors:
                y, x, l, t, r, b, q = tensor
                if q > qual[ind, y, x, 0]:
                    cls[ind, y, x, 0] = 1.0
                    left[ind, y, x, 0] = l
                    top[ind, y, x, 0] = t
                    right[ind, y, x, 0] = r
                    bottom[ind, y, x, 0] = b
                    qual[ind, y, x, 0] = q

        return inputs, cls, left, top, right, bottom

    def generate(self, annos, batch_size, verbose=False):
        processed_data = {}
        while True:
            batcher = self._get_processed_batches(annos, processed_data, batch_size)
            for size, processed_tensors in batcher:
                if verbose:
                    print 'next batch:'
                    for info in processed_tensors:
                        print 'img:', info['path']
                        print 'window: ', info['window']
                        print 'tensors:', info['tensors']

                inputs, cls, left, top, right, bottom = self._gen_tensors(size, processed_tensors)
                yield {
                    'input': inputs,
                    'block4_class': cls,
                    'dleft_block4_bbox': left,
                    'dtop_block4_bbox': top,
                    'dright_block4_bbox': right,
                    'dbottom_block4_bbox': bottom
                }


if __name__ == '__main__':

    BAR = 'Barcodes_1d'
    ROOT = open('../root').read().strip()
    print 'root', ROOT
    anno_path = os.path.join(ROOT, BAR, 'annotations.json')

    import json

    with open(anno_path) as f:
        anno = json.loads(f.read())
    print len(anno)
    print anno[0]

    from model import SSDModel

    ssd_model = SSDModel(do_not_load=True)

    ssd_generator = Generator(ROOT, ssd_model)
    gen = ssd_generator.generate(anno[:], 1, True)
    for i, t in enumerate(gen):
        print i, t
        if i > 100:
            break