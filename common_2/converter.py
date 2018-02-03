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
        with open(os.path.join(root, self.file_name)) as f:
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

                        center = rect.center()

                        max_shift = 2.0

                        dx = min((center.x - base_center.x) / (base_rect.width() + .0), max_shift)
                        dy = min((center.y - base_center.y) / (base_rect.height() + .0), max_shift)

                        max_stretch = 2.0

                        sh = math.log(min(rect.height() / (base_rect.height() + .0), max_stretch))
                        sw = math.log(min(rect.width() / (base_rect.width() + .0), max_stretch))

                        result.append(np.array([y, x, dy, dx, sh, sw]))
        return np.array(result)

    # get_overlapped_rects_info([Rect(9, 9, 22, 22)], 100, 100, 20, 20, 0.1, 1)

    def preprocess_rects(self, rects, model, height, width):

        if not isinstance(model, SSDModel):
            raise Exception("ssd model expected")

        result = []
        cnt = 0

        for scale, shape in zip(model.scales, model.shapes):

            curr = []

            for ratio in model.ratios:
                info = self._get_bblox_info(rects, height, width, shape[0], shape[1], scale, ratio)
                curr.append(info)
                cnt += len(info)

            result.append(curr)

        return np.array(result), cnt
        # print result

    def _restore_rects(self, tdy, tdx, tsh, tsw, cls, scale, ratio, threshold=None, top=None):
        # print ratio
        map_h, map_w = cls.shape[1:3]
        print map_h, map_w
        print cls.shape
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

                # print tdy.shape
                dy, dx, sh, sw = tdy[0, y, x, 0], tdx[0, y, x, 0], tsh[0, y, x, 0], tsw[0, y, x, 0]
                dy, dx, sh, sw = 0,0,0,0
                #sh, sw = 0, 0
                #dy, dx = 0, 0

                # print cls.shape
                conf = cls[0, y, x, 0]
                if conf > 0:
                    print conf
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

    def restore_rects(self, tensors, model, result, threshold=None, top=None):

        #print len(tensors)
        #print tensors.shape

        if not isinstance(model, SSDModel):
            raise Exception("ssd model expected")

        def cut_top(res):
            res = sorted(res, reverse=True, key=lambda val: val[0])
            if top is not None:
                res = res[:top]
            return res

        for j, (scale, ratio) in enumerate(zip(model.scales, model.ratios)):
            dy = tensors[4 * j + 0]
            dx = tensors[4 * j + 1]
            sh = tensors[4 * j + 2]
            sw = tensors[4 * j + 3]
            cls = tensors[4 * len(model.ratios) + j]

            # print cls[np.nonzero(cls)]

            result += self._restore_rects(dy, dx, sh, sw, cls, scale, ratio, threshold, top)
            print result

            if top is not None:
                result = cut_top(result)

        result = cut_top(result)
        return result

    def restore_rects_batch(self, tensors, model, threshold=None, top=None):

        result = []
        for tensor in tensors:
            result = self.restore_rects(tensor, model, result, threshold, top)

        confs, rects = zip(*result)
        return np.array(confs), np.array(rects)

    def process_rects(self, path, model, count=None, print_step=100):
        files = os.listdir(path)

        if count is not None:
            files = files[:count]

        processed_rects = {}
        print len(files), "files to process"
        for i, name in enumerate(files):

            rects_info = RectsInfo(name)
            rects_info.load_rects(path)

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
        print len(self.processed_rects), 'loaded'

    def add_empty_processed_rect(self, model, height=300, width=300):
        processed, cnt = self.preprocess_rects([], model, height, width)
        self.processed_rects[''] = processed


if __name__ == '__main__':
    ssd_model = SSDModel()
    ssd_converter = Converter('.')
    # ssd_converter.load('processed_rects.p')
    ssd_converter.process_rects('../Barcodes/Annotations', ssd_model, count=100, print_step=1)
    ssd_converter.add_empty_processed_rect(ssd_model)