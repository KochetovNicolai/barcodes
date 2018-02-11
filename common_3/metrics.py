from rect import Rect
import numpy as np


class RectWithConf:
    def __init__(self, rect, conf):
        self.rect = rect
        self.conf = conf


def strict_cover_strategy(rect, rects):
    for i, r in enumerate(rects):
        in_rect = rect.intersection(r.rect)
        out_rect = rect.union(r.rect)
        if in_rect.area() > 0.5 * out_rect.area():
            return i

    return len(rects)


def weak_cover_strategy(rect, rects):

    r_height = rect.height()
    r_width = rect.width()
    th_proportion = 4
    is_high_disproportional = th_proportion * r_width < r_height or th_proportion * r_height < r_width

    if not is_high_disproportional:
        return strict_cover_strategy(rect, rects)

    for i, r in enumerate(rects):
        in_rect = rect.intersection(r.rect)
        out_rect = rect.union(r.rect)
        if in_rect.area() > 0.5 * r.area() and in_rect.area() > 0.5 * out_rect.area() * 3 / th_proportion:
            return i

    return len(rects)


def auc11(true_rects, pred_rects, strategy=strict_cover_strategy):

    if len(true_rects) == 0:
        return 1.0

    sorted_rects = sorted(pred_rects, key=lambda x: x.conf, reverse=True)
    ps = [strategy(rect, sorted_rects) for rect in true_rects]
    recalls = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    precs = [0] * len(recalls)

    # cnts[i] = the number of detected true rects using first i+1 pred rects
    cnts = [0] * len(pred_rects)
    for p in ps:
        if p < len(cnts):
            cnts[p] += 1
    for i in range(1, len(pred_rects)):
        cnts[i] += cnts[i - 1]

    for i, cnt in enumerate(cnts):
        true_positive = cnts[i]
        recall = (0. + true_positive) / len(true_rects)
        prec = min((0. + true_positive) / (i + 1), 1.0)
        for j in range(len(recalls)):
            if recall >= recalls[j]:
                precs[j] = max(precs[j], prec)

    return np.mean(precs)


if __name__ == '__main__':
    print auc11([Rect(0, 0, 9, 9)], [RectWithConf(Rect(1, 1, 10, 10), 42)])
    print auc11([Rect(0, 0, 4, 4)], [RectWithConf(Rect(1, 1, 5, 5), 42)])
    print auc11([Rect(0, 0, 9, 9)], [RectWithConf(Rect(1, 1, 5, 5), 42), RectWithConf(Rect(1, 1, 10, 10), 41)])
    print auc11([Rect(0, 0, 9, 9)], [RectWithConf(Rect(1, 1, 5, 5), 41), RectWithConf(Rect(1, 1, 10, 10), 42)])
    print auc11([Rect(0, 0, 9, 9), Rect(20, 20, 40, 40)], [RectWithConf(Rect(1, 1, 10, 10), 42)])
    print auc11([Rect(0, 0, 9, 9), Rect(20, 20, 40, 40)], [RectWithConf(Rect(1, 1, 10, 10), 42),  RectWithConf(Rect(1, 1, 5, 5), 41)])
    print auc11([Rect(0, 0, 9, 9), Rect(20, 20, 40, 40)], [RectWithConf(Rect(1, 1, 10, 10), 41),  RectWithConf(Rect(1, 1, 5, 5), 42)])
    print auc11([Rect(0, 0, 9, 9), Rect(20, 20, 40, 40)], [RectWithConf(Rect(1, 1, 10, 10), 42),  RectWithConf(Rect(21, 21, 39, 39), 41)])
