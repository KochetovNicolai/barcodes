from rect import Point, Rect
import math
import numpy as np
import os
import pickle
from bs4 import BeautifulSoup
from model import SSDModel


class RectsInfo:
    def __init__(self, file_name):
        self.file_name = file_name
        self.height = None
        self.width = None
        self.rects = None

    def img_path(self):
        return os.path.splitext(self.file_name)[0] + '.jpg'

    def load_rects(self, root):
        fname = os.path.join(root, self.file_name)
        try:

            with open(fname) as f:
                text = f.read()
            # print text
            soup = BeautifulSoup(text, 'lxml')
            boxes = soup.find_all('bndbox')
            rects = []
            for box in boxes:
                l = int(box.find('xmin').text)
                r = int(box.find('xmax').text)
                t = int(box.find('ymin').text)
                b = int(box.find('ymax').text)
                # print l, t, r, b
                rects.append(Rect(l, t, r, b))

            self.rects = np.array(rects)

            size = soup.find('size')
            self.height = int(size.find('height').text)
            self.width = int(size.find('width').text)
        except Exception as e:
            print fname
            raise e


class Converter:
    def __init__(self, path):
        self.path = path
        self.processed_rects = None

    def _get_bblox_info(self, rects, img_h, img_w, map_h, map_w, scale, ratio):
        tmp = math.sqrt(ratio)
        # convert everything to img 0-1 axes
        rect_h = (scale / tmp)
        rect_w = (scale * tmp)
        # print "h, w", rect_h, rect_w

        result = []
        for y in range(0, map_h):
            for x in range(0, map_w):
                base_center = Point()
                base_center.y = (y + 0.5) / map_h
                base_center.x = (x + 0.5) / map_w
                base_rect = base_center.rect_from_center(rect_h, rect_w)
                # l, t, r, b = cx - rect_w_2, cy - rect_h_2, cx + rect_w_2, cy + rect_h_2
                for rect_real in rects:
                    rect = rect_real.copy()
                    rect.stretch(1.0 / img_h, 1.0 / img_w)
                    inner = rect.intersection(base_rect)
                    outer = rect.union(base_rect)

                    # r_l, r_t, r_r, r_b = np.array(rect, dtype=float)
                    # r_l, r_t, r_r, r_b = r_l / img_w, r_t / img_h, r_r / img_w, r_b / img_h
                    # i_l, i_t, i_r, i_b = max(l, r_l), max(t, r_t), min(r, r_r), min(b, r_b)
                    # o_l, o_t, o_r, o_b = min(l, r_l), min(t, r_t), max(r, r_r), max(b, r_b)
                    # print r_l, r_t, r_r, r_b
                    # print l, t, r, b
                    # print

                    r_height = rect.height()
                    r_width = rect.width()
                    is_same_oriented = (base_rect.width() <= base_rect.height()) == (r_width <= r_height)
                    th_proportion = 4
                    is_high_disproportional = th_proportion * r_width < r_height or th_proportion * r_height < r_width
                    th_weak_const = th_proportion * min(r_width, r_height) / max(r_width, r_height, 0.0001)  # < 1
                    weak_const = th_weak_const if is_same_oriented and is_high_disproportional else 1

                    '''
                    if is_same_oriented and is_high_disproportional:
                        if base_rect.width() > base_rect.height():
                            print '++++++++++'
                        print '################################'
                        print r_height, r_width
                        print th_weak_const, weak_const
                        print ',,,'
                        base_rect.dump()
                        rect.dump()
                        inner.dump()
                        outer.dump()
                        print inner.area(), outer.area()
                    '''

                    if inner.valid() and inner.area() > 0.5 * outer.area() * weak_const:

                        '''
                        if is_same_oriented and is_high_disproportional:
                            print '################################'
                            print r_height, r_width
                            print th_weak_const, weak_const
                            print ',,,'
                            base_rect.dump()
                            rect.dump()
                            inner.dump()
                            outer.dump()
                            print inner.area(), outer.area(), weak_const



                        
                        print '-----'
                        base_rect.dump()
                        rect.dump()
                        inner.dump()
                        outer.dump()
                        print inner.area(), outer.area()
                        '''

                        # if i_l < i_r and i_t < i_b:
                        # o_area = (o_r - o_l) * (o_b - o_t)
                        # i_area = (i_r - i_l) * (i_b - i_t)
                        # if i_area > o_area * 0.5:
                        # r_c_x = 0.5 * (r_l + r_r)
                        # r_c_y = 0.5 * (r_t + r_b)
                        center = rect.center()
                        # r_h = r_b - r_t
                        # r_w = r_r - r_l

                        max_shift = 2.0

                        dx = min((center.x - base_center.x) / (base_rect.width() + .0), max_shift)
                        dy = min((center.y - base_center.y) / (base_rect.height() + .0), max_shift)
                        # dy = (r_c_y - cy) / (2 * rect_h_2)
                        # dx = (r_c_x - cx) / (2 * rect_w_2)

                        max_stretch = 2.0

                        sh = math.log(min(rect.height() / (base_rect.height() + .0), max_stretch))
                        sw = math.log(min(rect.width() / (base_rect.width() + .0), max_stretch))
                        # sh = math.log(r_h / (2 * rect_h_2))
                        # sw = math.log(r_w / (2 * rect_w_2))
                        result.append(np.array([y, x, dy, dx, sh, sw]))
        return np.array(result)

    # get_overlapped_rects_info([Rect(9, 9, 22, 22)], 100, 100, 20, 20, 0.1, 1)

    def preprocess_rects(self, rects, model, height, width):

        if not isinstance(model, SSDModel):
            raise  Exception("ssd model expected")

        result = []
        cnt = 0
        for i, shape in enumerate(model.shapes):

            shape_h, shape_w = shape

            curr = []
            info = self._get_bblox_info(rects, height, width, shape_h, shape_w,
                                        math.sqrt(model.scales[i] * model.scales[i + 1]), 1)
            curr.append(info)
            cnt += len(info)

            for ratio in model.ratios:
                info = self._get_bblox_info(rects, height, width, shape_h, shape_w, model.scales[i], ratio)
                curr.append(info)
                cnt += len(info)

            result.append(curr)

        return np.array(result), cnt
        # print result

    def _restore_rects(self, mp, cls, scale, ratio, threshold=None, top=None):
        # print ratio
        map_h, map_w = mp.shape[:2]
        tmp = math.sqrt(ratio)
        rect_h = (scale / tmp)
        rect_w = (scale * tmp)
        # print "h, w", rect_h, rect_w

        def cut_top(res):
            res = sorted(res, reverse=True, key=lambda val: val[0])
            if top is not None:
                res = res[:top]
            return res

        res = []

        for y in range(0, map_h):
            for x in range(0, map_w):
                base_center = Point()
                base_center.y = (y + 0.5) / map_h
                base_center.x = (x + 0.5) / map_w

                dy, dx, sh, sw = mp[y, x].reshape((4,))
                dy, dx, sh, sw = 0,0,0,0
                #sh, sw = 0, 0
                #dy, dx = 0, 0

                sh = min(max(sh, -10), 10)
                sw = min(max(sw, -10), 10)

                conf = cls[y][x]
                if threshold is not None or conf > threshold:
                    r_x = dx * rect_w + base_center.x
                    r_y = dy * rect_h + base_center.y
                    r_w = rect_w * math.exp(sw)
                    r_h = rect_h * math.exp(sh)
                    center = Point()
                    center.x = r_x
                    center.y = r_y
                    res.append((conf, center.rect_from_center(r_h, r_w)))

                    if top is not None and len(res) > 2 * top:
                        res = cut_top(res)

        return cut_top(res)

    def restore_rects(self, mps, clss, model, threshold=None, top=None):

        if not isinstance(model, SSDModel):
            raise Exception("ssd model expected")

        def cut_top(res):
            res = sorted(res, reverse=True, key=lambda val: val[0])
            if top is not None:
                res = res[:top]
            return res

        result = []
        for j, (mp, cls) in enumerate(zip(mps, clss)):
            curr = []
            curr += self._restore_rects(mp[:, :, 0:4, :], cls[:, :, 0],
                                        math.sqrt(model.scales[j] * model.scales[j + 1]), 1, threshold, top)

            for i, ratio in enumerate(model.ratios):
                curr += self._restore_rects(mp[:, :, 4 * (i + 1):4 * (i + 2), :], cls[:, :, i + 1],
                                            model.scales[j], ratio, threshold, top)
            result += curr

            if top is not None:
                result = cut_top(result)

        result = cut_top(result)
        confs, rects = zip(*result)
        return np.array(confs), np.array(rects)

    def restore_rects_batch(self, tensors, model, threshold=None, top=None):
        mpss = tensors[:len(model.bbox_names)]
        clsss = tensors[len(model.bbox_names):]
        size = mpss[0].shape[0]
        #print size

        res = []
        for i in range(size):
            res.append(self.restore_rects([mps[i,:,:,:,:] for mps in mpss], [clss[i,:,:,:] for clss in clsss],
                                          model, threshold, top))
        return res

    def process_rects(self, path, model, count=None, print_step=100):
        files = os.listdir(path)

        if count is not None:
            files = files[:count]

        processed_rects = {}
        print len(files), "files to process"
        for i, name in enumerate(files):

            rects_info = RectsInfo(name)
            try:
                rects_info.load_rects(path)
            except:
                pass

            processed, cnt = self.preprocess_rects(rects_info.rects, model, rects_info.height, rects_info.width)
            processed_rects[name] = processed
            if print_step is not None and i % print_step == 0:
                print i, name, cnt

        self.processed_rects = processed_rects

    def dump(self, path):
        pickle.dump(self.processed_rects, open(path, "wb"))
        print len(self.processed_rects), 'dumped'

    def load(self, path):
        self.processed_rects = pickle.load(open(path, "rb"))
        for key, val in self.processed_rects.items():
            empt = True
            for rects in val:
                for rect in rects:
                    # print rect.shape
                    if len(rect):
                        empt = False
            if empt:
                del self.processed_rects[key]
        print len(self.processed_rects), 'loaded'

    def add_empty_processed_rect(self, model, height=300, width=300):
        processed, cnt = self.preprocess_rects([], model, height, width)
        self.processed_rects[''] = processed