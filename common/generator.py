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

    def _generate_for_bbox(self, processed_rects):
        all_bbox = []
        all_cls = []
        for rects, shape in zip(processed_rects, self.model.shapes):
            rat_bbox = []
            rat_cls = []
            for rect_lst in rects:
                bbox = np.zeros(list(shape) + [4])
                cls = np.zeros(list(shape) + [4])
                # print bbox.shape, cls.shape
                for rect in rect_lst:
                    y, x = map(int, rect[:2])
                    cls[y, x] = [1, 1, 1, 1]
                    bbox[y][x] = rect[2:]
                rat_bbox.append(bbox)
                rat_cls.append(cls)
            tmp_bbox = np.concatenate(rat_bbox, axis=2)
            tmp_cls = np.concatenate(rat_cls, axis=2)
            # print tmp_bbox.shape, tmp_cls.shape
            all_bbox.append(tmp_bbox)
            all_cls.append(tmp_cls)
        return np.array(all_bbox), np.array(all_cls)

    # { 'input' : (batch_size, height, width, depth)
    #   'bbox_name_{i}' : (batch_size, bbox_height, bbox_width, 4 * len(ratios), 1)
    #   'class_name_{i}' : (batch_size, bbox_height, bbox_width, len(ratios))
    def generate(self, processed_rects, batch_size, verbose=False):
        batch_x = []
        batch_y = [[] for _ in range(2 * len(self.model.shapes))]

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

                img = image.load_img(img_path, target_size=(300, 300))
                img = image.img_to_array(img)
                #img = np.expand_dims(img, axis=0)
                #img = preprocess_input(img)

                if verbose:
                    print img_path

                bboxes, classes = self._generate_for_bbox(processed_rects[processed_key])

                # add extra dimension for each bbox value
                rs = lambda x: np.reshape(x, tuple(list(x.shape) + [1]))

                bboxes_tensors = [rs(np.append(bbox, cls, axis=2)) for bbox, cls in zip(bboxes, classes)]
                classes_tensors = [rs(cls[:, :, ::4]) for cls in classes]
                y = bboxes_tensors + classes_tensors

                batch_x.append(np.array(img))
                for i in range(len(y)):
                    batch_y[i].append(y[i])

                # create Numpy arrays of input data and labels, from each line in the file
                # x, y = process_line(line)
                if len(batch_x) == batch_size:
                    batch_y = list(map(np.array, batch_y))
                    res = {name: value for name, value in zip(self.model.bbox_names + self.model.cls_names, batch_y)}
                    # res['input_1'] = np.array(batch_x)
                    yield ({'input': preprocess_input(np.array(batch_x))}, res)
                    batch_x = []
                    batch_y = [[] for _ in range(2 * len(self.model.shapes))]