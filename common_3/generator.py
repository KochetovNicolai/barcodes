import numpy as np
import os
from model import SSDModel, PixelsModel
from rect import Rect
import random
import skimage.transform

from scipy.misc import imresize

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import tensorflow as tf

class NPArrayCache:

    class Cell:
        def __init__(self, arr, time):
            self.arr = arr
            self.time = time

    def __init__(self, mem_limit_bytes):
        self.cache = {}
        self.tot_bytes = 0
        self.lre_queue = []
        self.time = 0
        self.mem_limit_bytes = mem_limit_bytes

    def get(self, name):
        if name not in self.cache:
            return None

        self.cache[name].time = self.time
        self.lre_queue.append((name, self.time))
        self.time += 1
        return self.cache[name].arr

    def set(self, name, arr):
        self.cache[name] = NPArrayCache.Cell(arr, self.time)
        self.lre_queue.append((name, self.time))
        self.time += 1
        self.tot_bytes += arr.nbytes
        self.clean()

    def clean(self):
        if self.tot_bytes < self.mem_limit_bytes:
            return

        while self.tot_bytes > self.mem_limit_bytes and len(self.lre_queue):
            name, time = self.lre_queue[0]
            self.lre_queue = self.lre_queue[1:]
            if time == self.cache[name].time:
                self.tot_bytes -= self.cache[name].arr.nbytes
                del self.cache[name]

        if self.tot_bytes > self.mem_limit_bytes:
            raise Exception("Logical error: queue is empty, but tot_bytes > mem_limit_bytes ({} > {}), cache_size = {}"
                            .format(self.tot_bytes, self.mem_limit_bytes, len(self.cache)))


class Generator:

    def __init__(self, root, model, cache_mem_limit_bytes=1 << 30):
        if not isinstance(model, SSDModel) and not isinstance(model, PixelsModel):
            raise Exception('expected SSDModel or PixelsModel')
        self.root = root
        self.model = model
        if isinstance(model, SSDModel):
            self.max_ratio = model.max_ratio
            self.num_poolings = model.num_poolings
        else:
            self.pref_rect_short_log_size = 6
        self.cache = NPArrayCache(cache_mem_limit_bytes)

    def _get_possible_windows(self, h, w):
        mn = min(h, w)
        mx = max(h, w)
        mx = min(mx, int(mn * self.max_ratio))
        window = 1 << max(mn.bit_length(), self.num_poolings)
        res = []
        while window < 2 * mx:
            if window > mn:
                res.append(window)
            window *= 2

        res.append(window)
        return res

    def _get_possible_pixel_factors(self, h, w):
        mn = min(h, w)
        bit_len = mn.bit_length()

        min_factor = max(bit_len - self.pref_rect_short_log_size, 0)
        max_factor = max(1 + bit_len - self.pref_rect_short_log_size, 0)
        if min_factor == max_factor:
            return [min_factor]
        return [min_factor, max_factor]

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

    def _get_rect_tensor_descs(self, rect, window_rect, factor):
        assert isinstance(rect, Rect)
        assert isinstance(window_rect, Rect)

        window_size = window_rect.height()
        map_size = window_size >> (factor + self.num_poolings)

        edge = window_size / map_size

        center = rect.center()
        sx = int(center.x - window_rect.left)
        sy = int(center.y - window_rect.top)
        x = int(sx / edge)
        y = int(sy / edge)

        res = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                wx = x + dx
                wy = y + dy
                if 0 <= wx and wx + 1 < map_size and 0 <= wy and wy + 1 < map_size:
                    # window coord
                    wrect = Rect(wx * edge, wy * edge, (wx + 2) * edge, (wy + 2) * edge)
                    # img coord
                    wrect.move(window_rect.top, window_rect.left)

                    i = wrect.intersection(rect)

                    if i.valid(): #self._check_rect(rect, wrect):
                        # img coord
                        #i = wrect.intersection(rect)
                        qual_out = i.area() / float(wrect.area())
                        # if qual_out <= 0:
                        #     print rect ,wrect, i
                        # window coord
                        i.move(-wrect.top, -wrect.left)
                        # window 0..1 coord
                        i.stretch(1.0 / wrect.height(), 1.0 / wrect.width())

                        res.append((wy, wx, i.left, i.top, i.right, i.bottom, qual_out))

        return res

    def _get_rect_tensor_descs_no_overlap(self, rect, window_rect, factor):
        assert isinstance(rect, Rect)
        assert isinstance(window_rect, Rect)

        window_size = window_rect.height()
        map_size = window_size >> (factor + self.num_poolings)

        edge = window_size / map_size

        center = rect.center()
        sx = int(center.x - window_rect.left)
        sy = int(center.y - window_rect.top)
        x = int(sx / edge)
        y = int(sy / edge)

        res = []

        for dx in range(-3, 3):
            for dy in range(-2, 2):
                wx = x + dx
                wy = y + dy
                if 0 <= wx and wx < map_size and 0 <= wy and wy < map_size:
                    # window coord
                    wrect = Rect(wx * edge, wy * edge, (wx + 1) * edge, (wy + 1) * edge)
                    # img coord
                    wrect.move(window_rect.top, window_rect.left)

                    i = wrect.intersection(rect)

                    if i.valid(): #self._check_rect(rect, wrect):
                        # img coord
                        #i = wrect.intersection(rect)
                        qual_out = i.area() / float(wrect.area())
                        # if qual_out <= 0:
                        #     print rect ,wrect, i
                        # window coord
                        i.move(-wrect.top, -wrect.left)
                        # window 0..1 coord
                        i.stretch(1.0 / wrect.height(), 1.0 / wrect.width())

                        if qual_out > 0.01:
                            res.append((wy, wx, i.left, i.top, i.right, i.bottom, qual_out))

        return res

    def _create_window_for_rect(self, rect, window, img_h, img_w):
        assert isinstance(rect, Rect)
        # if window > img_h or window > img_w:
        #     return None
        #
        # assert rect.height() <= img_h and rect.width() <= img_w
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

    def _process_ssd_anno(self, anno, rects, window_rect, factor):

        # print 'size', window_rect.width(), 'factor', factor, 'num_p', self.num_poolings, 'cnt'

        tensor_descs = []
        for rect in rects:
            tensor_descs += self._get_rect_tensor_descs_no_overlap(rect, window_rect, factor)

        if not tensor_descs:
            return None

        return {
            'path': os.path.join(anno['path'], anno['name']),
            'shape': anno['shape'],
            'window': window_rect,
            'tensors': tensor_descs
        }

    def _process_empty_anno(self, anno, window_rect):
        return {
            'path': os.path.join(anno['path'], anno['name']),
            'shape': anno['shape'],
            'window': window_rect,
            'tensors': [],
            'rects': []
        }


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
                factor = 0
                while factor == 0 or (window << factor) <= min(img_h, img_w):

                    window_rect = self._create_window_for_rect(rect, window << factor, img_h, img_w)
                    if window_rect is None:
                        continue

                    desc = self._process_ssd_anno(anno, rects, window_rect, factor)
                    if desc is not None:

                        if 1 << factor not in processed_data:
                            processed_data[1 << factor] = []

                        processed_data[1 << factor].append(desc)

                    factor += 1

    def _process_anno_2(self, anno, processed_data, desc_processor):

        path = os.path.join(anno['path'], anno['name'])
        if not os.path.exists(os.path.join(self.root, path)):
            #print path
            return

        rects = list(map(lambda r: Rect(r[0], r[2], r[1], r[3]), anno['Rects']))
        variants = []
        best_min_side = 64

        img_h, img_w, depth = anno['shape']
        for rect in rects:
            # factors = self._get_possible_pixel_factors(rect.height(), rect.width())
            best_variant, best_fine = None, None
            rect_h, rect_w = rect.height(), rect.width()
            rect_min = min(rect_h, rect_w)

            for factor in range(10): # factors:

                curr_min = rect_min >> factor
                curr_fine = abs(curr_min - best_min_side)

                log_window = factor + 9 # 512 x 512 #self.pref_rect_short_log_size
                if (1 << log_window) <= max(img_h, img_w): #while

                    if best_fine is None or curr_fine < best_fine:
                        best_fine = curr_fine

                        size = 1 << (log_window - factor)
                        window_rect = self._create_window_for_rect(rect, 1 << log_window, img_h, img_w)

                        desc = desc_processor(self, anno, rects, window_rect, factor)
                        if desc is not None:
                            best_variant = (size, desc)

                    log_window += 1

            if best_variant is not None:
                variants.append(best_variant)

        num_variants = len(variants)
        if num_variants == 0:
            # anno is empty. get any rect
            x, y = (img_w >> 1), (img_h >> 1)
            max_log = min(12, int(max(img_h, img_w, 512)).bit_length())
            scale = random.randint(9, max_log + 1)
            window_rect = self._create_window_for_rect(Rect(x,y,x,y), 512 << (scale-9), img_h, img_w)

            desc = self._process_empty_anno(anno, window_rect)
            variants.append((512, desc))
            num_variants = 1

        # find any
        ind = random.randint(0, num_variants - 1)
        size, desc = variants[ind]

        if size not in processed_data:
            processed_data[size] = []

        processed_data[size].append(desc)

    def _process_anno_pixels(self, anno, processed_data):

        path = os.path.join(anno['path'], anno['name'])
        if not os.path.exists(os.path.join(self.root, path)):
            #print path
            return

        rects = list(map(lambda r: Rect(r[0], r[2], r[1], r[3]), anno['Rects']))

        variants = []

        best_min_side = 64

        img_h, img_w, depth = anno['shape']
        for rect in rects:
            # factors = self._get_possible_pixel_factors(rect.height(), rect.width())
            best_variant = None
            best_fine = None
            rect_h = rect.height()
            rect_w = rect.width()
            rect_min = min(rect_h, rect_w)

            for factor in range(10): # factors:

                curr_min = rect_min >> factor
                curr_fine = abs(curr_min - best_min_side)

                log_window = factor + 9 # 512 x 512 #self.pref_rect_short_log_size
                if (1 << log_window) <= max(img_h, img_w): #while

                    if best_fine is None or curr_fine < best_fine:
                        best_fine = curr_fine

                        size = 1 << (log_window - factor)
                        window_rect = self._create_window_for_rect(rect, 1 << log_window, img_h, img_w)

                        desc = {
                            'path': path,
                            'shape': anno['shape'],
                            'window': window_rect,
                            'rects': rects
                        }

                        best_variant = (size, desc)

                    log_window += 1

            if best_variant is not None:
                variants.append(best_variant)

        cnt = len(variants)
        if cnt == 0:
            # anno is empty. get any rect
            x, y = (img_w >> 1) , (img_h >> 1)
            max_log = min(12, int(max(img_h, img_w, 512)).bit_length())
            scale = random.randint(9, max_log + 1)
            window_rect = self._create_window_for_rect(Rect(x,y,x,y), 512 << (scale-9), img_h, img_w)

            desc = {
                'path': os.path.join(anno['path'], anno['name']),
                'shape': anno['shape'],
                'window': window_rect,
                'rects': rects
            }
            variants.append((512, desc))
            cnt = 1

        # find any
        ind = random.randint(0, cnt - 1)
        size, desc = variants[ind]

        if size not in processed_data:
            processed_data[size] = []

        processed_data[size].append(desc)

    def _gen_pixels_tensor(self, size, infos):
        batch_size = len(infos)
        rands = np.random.random(batch_size)

        def test_copy(z, img, h, w):
            z[:h, :w, :] = img[:h, :w, :]

        # input tensor
        inputs = []
        for i, info in enumerate(infos):
            window = info['window']

            path = os.path.join(self.root, info['path'])
            img = self.cache.get(path)
            if img is None:
                img = image.img_to_array(image.load_img(path, grayscale=True))
                self.cache.set(path, img)

            img = img[window.top:window.bottom, window.left:window.right, :] / 255.0
            z = np.ones((window.height(), window.width(), 1), dtype=float)
            test_copy(z, img, img.shape[0], img.shape[1])
            img = z

            img = skimage.transform.resize(img, (size, size), mode='reflect')
            if rands[i] < 0.5:
                img = 1.0 - img
            inputs.append(img)

        #inputs = preprocess_input(np.array(inputs))
        inputs = np.array(inputs)

        borders = False

        # output tensor
        outputs = np.zeros((batch_size, size, size, 1))
        # if borders:
        #     outputs = np.zeros((batch_size, size, size, 1))
        # else:
        #     outputs = np.zeros((batch_size, size, size, 2))
        #     outputs[:, :, :, 0] = 1.0

        for i, info in enumerate(infos):
            window = info['window']
            rects = info['rects']
            factor = window.width() / size
            assert isinstance(window, Rect)
            for rect in rects:
                inter = window.intersection(rect)
                if inter.valid():
                    inter.move(-window.top, -window.left)
                    inter.stretch(1.0 / factor, 1.0 / factor)
                    inter.integerify()

                    if not borders:
                        #outputs[i, int(inter.top):int(inter.bottom), int(inter.left):int(inter.right), 1] = 0.0
                        outputs[i, int(inter.top):int(inter.bottom), int(inter.left):int(inter.right), 0] = 1.0
                    else:
                        # exp(-dist(the nearest border))
                        inter_height = inter.height()
                        inter_width = inter.width()
                        ind = np.indices((inter_height, inter_width))
                        exp_1 = 1.0 / (1.0 + ind[0])
                        exp_2 = 1.0 / (1.0 + ind[1])
                        #print exp_1.shape
                        #print exp_2.shape
                        filler = np.maximum(exp_1, exp_2)
                        filler = np.maximum(filler, filler[::-1, ::-1])
                        outputs[i, inter.top:inter.bottom, inter.left:inter.right, 0] = filler

        do_augmentation = True
        if do_augmentation:
            conc = np.concatenate((inputs, outputs), axis=-1)
            #print conc.shape
            augm = image.ImageDataGenerator(rotation_range=15.,height_shift_range=0.1, width_shift_range=0.1,
                                            fill_mode='constant', cval=0.0)
            aug_in, aug_out = next(augm.flow(conc, np.ones(batch_size), batch_size=batch_size))
            #print aug_in.shape
            inputs, outputs = aug_in[:,:,:,:1], aug_in[:,:,:,1:]
            #print inputs.shape

        out_shaped = np.zeros((batch_size, 16, 16, 2))
        for i in range(0, batch_size):
            out_shaped[i,:,:,1:] = skimage.transform.resize(outputs[i], (16, 16), mode='reflect')
            out_shaped[i,:,:,0] = 1.0 - out_shaped[i,:,:,1]
        #outputs = tf.image.resize_images(outputs, (h >> 5, w >> 5)).eval(session=self.sess)

        return inputs, out_shaped

    def _get_processed_batches(self, processor, annos, processed_data, max_subimg_size, permute):

        if permute:
            indexes = np.random.permutation(len(annos))
        else:
            indexes = np.arange(len(annos))

        for index in indexes:
            anno = annos[index]
            processor(self, anno, processed_data)

            if False:
                for key, value in processed_data.items():
                    print (key, ':', len(value))

            for key in processed_data.keys():
                val = processed_data[key]

                divisor = max(1, key) ** 2

                if divisor > max_subimg_size:
                    # not enough memory for subimage
                    processed_data[key] = []
                    continue

                batch_size = int(max_subimg_size / divisor)

                while len(val) >= batch_size:
                    yield key, val[:batch_size]
                    val = val[batch_size:]

                processed_data[key] = val

                if False:
                    for key2, value2 in processed_data.items():
                        print (key2, ':', len(value2))

    def _fill_bboxes(self, img_size, batch_size, processed_tensors):
        # output_tensors
        size = img_size >> self.num_poolings
        bbox_size = size
        cls = np.zeros((batch_size, bbox_size, bbox_size, 2))
        cls[:,:,:,0] = np.ones((batch_size, bbox_size, bbox_size))
        qual = np.zeros((batch_size, bbox_size, bbox_size, 1))
        lr = np.zeros((batch_size, bbox_size, bbox_size, 2))
        tb = np.zeros((batch_size, bbox_size, bbox_size, 2))

        for ind, info in enumerate(processed_tensors):
            tensors = info['tensors']
            for tensor in tensors:
                y, x, l, t, r, b, q = tensor
                if q > qual[ind, y, x, 0]:
                    cls[ind, y, x, 0] = 0.0
                    cls[ind, y, x, 1] = 1.0
                    lr[ind, y, x, 0] = l
                    lr[ind, y, x, 1] = 1.0 - r
                    tb[ind, y, x, 0] = t
                    tb[ind, y, x, 1] = 1.0 - b
                    qual[ind, y, x, 0] = q

        return cls, lr, tb

    def _gen_tensors(self, img_size, processed_tensors):
        #print 'img_size', img_size
        batch_size = len(processed_tensors)

        size = img_size >> self.num_poolings

        def test_copy(z, img, h, w):
            z[:h, :w, :] = img[:h, :w, :]

        # input tensor
        inputs = []
        for info in processed_tensors:
            window = info['window']

            path = os.path.join(self.root, info['path'])
            img = self.cache.get(path)
            if img is None:
                img = image.img_to_array(image.load_img(path))
                self.cache.set(path, img)

            img = img[window.top:window.bottom, window.left:window.right, :]
            z = np.zeros((window.height(), window.width(), 3), dtype=float)
            test_copy(z, img, img.shape[0], img.shape[1])
            img = z

            img = skimage.transform.resize(img, (img_size, img_size), mode='reflect')
            inputs.append(img)
        inputs = preprocess_input(np.array(inputs))

        cls, lr, tb = self._fill_bboxes(img_size, batch_size, processed_tensors)

        return inputs, cls, lr, tb

    def generate(self, annos, max_subimg_size, verbose=False, permute=True):
        processed_data = {}

        def wrap(bbox, cls):
            return np.concatenate((bbox, cls[:,:,:,-1:]), axis=-1)

        def processor(self, anno, processed_data):
            return Generator._process_anno_2(self, anno, processed_data, Generator._process_ssd_anno)

        while True:

            batcher = self._get_processed_batches(processor, annos, processed_data, max_subimg_size, permute)
            for size, processed_tensors in batcher:
                if verbose:
                    print ('next batch:')
                    for info in processed_tensors:
                        print ('img:', info['path'])
                        print ('window: ', info['window'])
                        print ('tensors:', info['tensors'])

                inputs, cls, lr, tb = self._gen_tensors(size, processed_tensors)
                yield {'input': inputs}, {
                    'class': cls,
                    'lr_bbox': wrap(lr, cls),
                    'tb_bbox': wrap(tb, cls),
                }

    def generate_pixels(self, annos, max_batch_size_for_smallest_image, verbose=False, permute=True):
        processed_data = {}

        while True:
            batcher = self._get_processed_batches(Generator._process_anno_pixels, annos,
                                                  processed_data, max_batch_size_for_smallest_image, permute)
            for size, processed_tensors in batcher:
                if verbose:
                    print ('next batch:')
                    for info in processed_tensors:
                        print ('img:', info['path'])
                        print ('window: ', info['window'])

                inputs, outputs = self._gen_pixels_tensor(size, processed_tensors)
                yield inputs, outputs


if __name__ == '__main__':
    ROOT = open('../root').read().strip()
    BAR = 'Barcodes_1d'
    path = os.path.join(ROOT, 'datasets', 'any_dataset')
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
              os.path.splitext(f)[1] == '.tif']

    print len(images)

    from model import SSDModel

    ssd_model = SSDModel()
    ssd_generator = Generator(ROOT, ssd_model)

    import json

    anno_path = os.path.join(ROOT, BAR, 'annotations.json')
    with open(anno_path) as f:
        anno = json.loads(f.read())
    print (len(anno))
    print (anno[0])

    generator = ssd_generator.generate(anno, 1 * 512 * 512, True)
    gen = next(generator)


    # BAR = 'Barcodes_1d'
    # ROOT = open('../root').read().strip()
    # print ('root', ROOT)
    # anno_path = os.path.join(ROOT, BAR, 'annotations.json')
    #
    # import json
    #
    # with open(anno_path) as f:
    #     anno = json.loads(f.read())
    # print (len(anno))
    # print (anno[0])
    #
    # from model import SSDModel
    #
    # ssd_model = SSDModel(do_not_load=True)
    #
    # ssd_generator = Generator(ROOT, ssd_model, cache_mem_limit_bytes=4 << 30)
    # gen = ssd_generator.generate(anno[:], 1024, False)
    # for i, t in enumerate(gen):
    #     # print i, t
    #     if i > 100:
    #         break


# Barcodes_1d/Upc-A 3rdScanWithTextHeavyDamaged/0028.jpg
#
# export LC_ALL="en_US.UTF-8"
# export LC_CTYPE="en_US.UTF-8"
# sudo dpkg-reconfigure locales