from PIL import Image
import numpy as np
import os
from model import SSDModel

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

class Generator:

    def __init__(self, path, path_empty, model, train, empty_ratio=0.5):
        if not isinstance(model, SSDModel):
            raise Exception('expected SSDModel')
        self.model = model
        self.path = path
        self.path_empty = path_empty
        self.train = set(train) if train is not None else None
        self.empty_ratio = empty_ratio

    def _generate_for_bbox(self, rects, shapes):

        all_bbox = []
        all_cls = []

        for rect_lst_list, shape in zip(rects, shapes):

            cur_bbox = []
            cur_cls = []
            # print bbox.shape, cls.shape
            for rect_list in rect_lst_list:
                dy, dx, sh, sw = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
                cls = np.zeros(shape)
                for rect in rect_list:
                    y, x = map(int, rect[:2])
                    cls[y, x] = 1
                    dy[y][x], dx[y][x], sh[y][x], sw[y][x] = rect[2:]
                cur_bbox += [dy, dx, sh, sw]
                cur_cls += [cls, cls, cls, cls]

            all_bbox.append(cur_bbox)
            all_cls.append(cur_cls)

        return all_bbox, all_cls

    # { 'input' : (batch_size, height, width, depth)
    #   'bbox_name_{i}' : (batch_size, bbox_height, bbox_width, 4 * len(ratios), 1)
    #   'class_name_{i}' : (batch_size, bbox_height, bbox_width, len(ratios))
    def generate(self, processed_rects, batch_size, verbose=False):
        batches_x = []
        batches_y = []

        for i in range(len(self.model.shapes)):
            batches_x.append([])
            # (dy, dx, dh, dw, cls) * 5 ratios
            batches_y.append([[] for _ in range(5 * len(self.model.ratios))])

        tmp_cnt = 0

        while 1:

            keys = np.array(processed_rects.keys())
            ind = np.arange(len(keys))
            np.random.shuffle(ind)
            rnd = np.random.rand(len(keys))
            empty_name = os.listdir(self.path_empty)
            empty_ind = np.random.randint(len(empty_name), size=len(keys))

            for jj in ind:
                # print jj, rnd[jj]
                if rnd[jj] < self.empty_ratio:
                    processed_key = ''
                else:
                    processed_key = keys[jj]

                #print processed_key

                if processed_key != '' and self.train and processed_key not in self.train:
                    continue

                name = os.path.splitext(processed_key)[0]

                if processed_key == '':
                    img_path = os.path.join(self.path_empty, empty_name[empty_ind[jj]])
                else:
                    img_path = os.path.join(self.path, name + '.jpg')

                if verbose:
                    print img_path

                all_bbox, all_cls = self._generate_for_bbox(processed_rects[processed_key], self.model.shapes)

                for i, (cur_bbox, cur_cls, divisor) in enumerate(zip(all_bbox, all_cls, self.model.divisors)):

                    bboxes_tensors = [np.stack((bbox, cls), axis=-1) for bbox, cls in zip(cur_bbox, cur_cls)]
                    y = bboxes_tensors + [np.expand_dims(cls, axis=-1) for cls in cur_cls[::4]]

                    shape = (300 / divisor, 300 / divisor)

                    img = image.load_img(img_path, target_size=shape)
                    img = image.img_to_array(img)

                    batches_x[i].append(img)
                    for j in range(len(y)):
                        batches_y[i][j].append(y[j])

                tmp_cnt += 1
                #print tmp_cnt

                if tmp_cnt > 100:
                    print batches_x
                    print batches_y

                    print np.array(batches_x).shape
                    print np.array(batches_y).shape
                    raise Exception('trigger')

                # create Numpy arrays of input data and labels, from each line in the file
                # x, y = process_line(line)
                if len(batches_x[0]) == batch_size:
                    for batch_x, batch_y in zip(batches_x, batches_y):
                        batch_y = list(map(np.array, batch_y))
                        res = {name: value for name, value in zip(self.model.bbox_names + self.model.cls_names, batch_y)}
                        # res['input_1'] = np.array(batch_x)
                        yield ({'input': preprocess_input(np.array(batch_x))}, res)

                    batches_x = []
                    batches_y = []

                    for i in range(len(self.model.shapes)):
                        batches_x.append([])
                        # (dy, dx, dh, dw, cls) * 5 ratios
                        batches_y.append([[] for _ in range(5 * len(self.model.ratios))])

                    tmp_cnt = 0