from rect import Point, Rect
import math
import numpy as np
import os
import pickle
from model import SSDModel
import json


class Converter:
    def __init__(self, model, verbose=False):

        if not isinstance(model, SSDModel):
            raise Exception("ssd model expected")

        self.model = model
        self.verbose = verbose

    def _restore_rects(self, lr, tb, cls, num_poolings, threshold=None, top=None):
        # print ratio
        map_h, map_w = lr.shape[1:3]
        if self.verbose:
            print map_h, map_w
            if cls is not None:
                print cls.shape

        left_logit = lr[:,:,:,0]
        right_logit = lr[:,:,:,1]
        top_logit = tb[:,:,:,0]
        bottom_logit = tb[:,:,:,1]

        if cls is not None:
            bg_logit = np.exp(cls[:,:,:,0])
            bar_logit = np.exp(cls[:,:,:,1])
            prob = np.exp(bar_logit) / (np.exp(bar_logit) + np.exp(bg_logit))

        dleft = left_logit
        dright = 1.0 - right_logit
        dtop = top_logit
        dbottom = 1.0 - bottom_logit

        def cut_top(res):
            res = sorted(res, reverse=True, key=lambda val: val[0])
            if top is not None:
                res = res[:top]
            return res

        res = []

        for y in range(0, map_h):
            for x in range(0, map_w):

                dl, dr, dt, db = dleft[0, y, x], dright[0, y, x], dtop[0, y, x], dbottom[0, y, x]

                #dy, dx, sh, sw = 0,0,0,0
                #sh, sw = 0, 0
                #dy, dx = 0, 0

                # print cls.shape
                conf = prob[0, y, x] if cls is not None else 0

                if self.verbose and conf > 0:
                    print conf, y, x, dl, dr, dt, db
                if threshold is not None or conf > threshold:
                    rect = Rect(x + dl * 2, y + dt * 2, x + dr * 2, y + db * 2)
                    rect.stretch(1 << (num_poolings), 1 << (num_poolings))

                    if self.verbose:
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

        lr, tb, cls = tensors

        result = self._restore_rects(lr, tb, cls, self.model.num_poolings, threshold, top)
        if self.verbose:
            print result

        result = cut_top(result)
        return tuple(r[0] for r in result), tuple(r[1] for r in result)


if __name__ == '__main__':
    ssd_model = SSDModel()
