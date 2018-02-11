from model import SSDModel
from converter import Converter
import numpy as np
from PIL import Image

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from metrics import RectWithConf


class Predictor:
    def __init__(self, ssd_model, ssd_converter):
        self.ssd_model = ssd_model
        self.ssd_converter = ssd_converter

    def predict(self, path, top=10, threshold=None):

        res = []

        for divisor in self.ssd_model.divisors:

            shape = (300 / divisor, 300 / divisor)

            img = image.load_img(path, target_size=shape)
            img = image.img_to_array(img)
            img = img.reshape([1] + list(img.shape))
            img = preprocess_input(img)

            tensor = self.ssd_model.model.predict(img)
            res2 = []
            res2 = self.ssd_converter.restore_rects(tensor, self.ssd_model, res2, top=top, threshold=threshold)
            res += res2
            #print res

        #print 42
        #print res
        confs, rects = [r[0] for r in res], [r[1] for r in res]
        print confs
        print rects
        for rect in rects:
            rect.stretch(300, 300)
        return sorted([RectWithConf(r, c) for r, c in zip(rects, confs)], key=lambda x: x.conf, reverse=True)
