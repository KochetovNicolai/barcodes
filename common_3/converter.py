from rect import Point, Rect
import math
import numpy as np
import os
import pickle
from model import SSDModel
import json


class Converter:
    def __init__(self, model):

        if not isinstance(model, SSDModel):
            raise Exception("ssd model expected")

        self.model = model

    def _restore_rects(self, dleft, dtop, dright, dbottom, cls, num_poolings, threshold=None, top=None):
        # print ratio
        map_h, map_w = cls.shape[1:3]
        cnt_h = (map_h + 1) / 2
        cnt_w = (map_w + 1) / 2
        print map_h, map_w
        print cnt_h, cnt_w
        print cls.shape

        def cut_top(res):
            res = sorted(res, reverse=True, key=lambda val: val[0])
            if top is not None:
                res = res[:top]
            return res

        res = []

        for y in range(0, map_h):
            for x in range(0, map_w):

                # print tdy.shape
                dl, dr, dt, db = dleft[0, y, x, 0], dright[0, y, x, 0], dtop[0, y, x, 0], dbottom[0, y, x, 0]

                #dy, dx, sh, sw = 0,0,0,0
                #sh, sw = 0, 0
                #dy, dx = 0, 0

                # print cls.shape
                conf = cls[0, y, x, 0]
                if conf > 0:
                    print conf, y, x, dl, dr, dt, db
                if threshold is not None or conf > threshold:
                    rect = Rect(x + dl * 2, y + dt * 2, x + dr * 2, y + db * 2)
                    print rect
                    rect.stretch(1 << (num_poolings), 1 << (num_poolings))
                    print rect
                    res.append((conf, rect))

                    if top is not None and len(res) > 2 * top:
                        res = cut_top(res)

        return cut_top(res)

    def restore_rects(self, tensors, threshold=None, top=None):

        #print len(tensors)
        #print tensors.shape

        def cut_top(res):
            res = sorted(res, reverse=True, key=lambda val: val[0])
            if top is not None:
                res = res[:top]
            return res

        dleft, dtop, dright, dbottom, cls = tensors

        result = self._restore_rects(dleft, dtop, dright, dbottom, cls, self.model.num_poolings, threshold, top)
        print result

        result = cut_top(result)
        return tuple(r[0] for r in result), tuple(r[1] for r in result)


if __name__ == '__main__':
    ssd_model = SSDModel()
