from model import SSDModel
from metrics import auc11
from converter import RectsInfo
import os


class Evaluator:
    def __init__(self, predictor):
        self.predictor = predictor

    def evaluate(self, img_root, conf_root, names, top=10, metric=auc11, verbose=100):

        print img_root
        total_qual = 0.0

        cnt = 0

        def check(rect):
            return rect.height() < 4 * rect.width() and rect.width() < 4 * rect.height()

        for i, name in enumerate(names):
            info = RectsInfo(name)
            try:
                info.load_rects(conf_root)
            except Exception as e:
                print e
                continue

            rects = list(filter(check, info.rects))
            if len(rects) == 0 or len(rects) < len(info.rects):
                continue

            img_name = os.path.join(img_root, os.path.splitext(info.file_name)[0]) + '.jpg'
            # print img_name
            pred = self.predictor.predict(img_name, top=top, threshold=None)
            qual = metric(rects, pred)
            total_qual += qual

            cnt += 1

            if verbose and cnt % verbose == 0:
                print img_name
                print cnt, 'avg', total_qual / cnt, 'last', qual

        return total_qual / cnt
