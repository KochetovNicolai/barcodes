import keras
import os
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Reshape, Activation
from keras.applications.vgg16 import VGG16
import math
import numpy as np


def get_ssd_model_description(count=6):
    def reshaped(layer, name):
        shape = tuple(list(layer._keras_shape)[1:] + [1])
        print shape
        #return Reshape(shape, name=name)(layer)
        return Activation('tanh', name=name)(Reshape(shape, name=name + '_reshape')(layer))

    def reshaped_cls(layer, name):
        shape = tuple(list(layer._keras_shape)[1:] + [1])
        print shape
        return Activation('sigmoid', name=name)(Reshape(shape, name=name + '_reshape')(layer))

    img_input = Input(shape=(300, 300, 3), name='input')

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_bbox = reshaped(Conv2D(4 * count, (3, 3), padding='same', name='block4_conv_bbox_0')(x), 'block4_bbox')
    block4_conv_class = reshaped_cls(Conv2D(count, (3, 3), padding='same', name='block4_conv_class_0')(x), 'block4_class')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_bbox = reshaped(Conv2D(4 * count, (3, 3), padding='same', name='block5_conv_bbox_0')(x), 'block5_bbox')
    block5_conv_class = reshaped_cls(Conv2D(count, (3, 3), padding='same', name='block5_conv_class_0')(x), 'block5_class')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Block 6
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block6_conv1')(x)
    block6_bbox = reshaped(Conv2D(4 * count, (3, 3), padding='same', name='block6_conv_bbox_0')(x), 'block6_bbox')
    block6_conv_class = reshaped_cls(Conv2D(count, (3, 3), padding='same', name='block6_conv_class_0')(x), 'block6_class')
    # Block 7
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block7_conv1')(x)
    block7_bbox = reshaped(Conv2D(4 * count, (3, 3), padding='same', name='block7_conv_bbox_0')(x), 'block7_bbox')
    block7_conv_class = reshaped_cls(Conv2D(count, (3, 3), padding='same', name='block7_conv_class_0')(x), 'block7_class')
    # Block 8
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block8_conv1')(x)
    block8_bbox = reshaped(Conv2D(4 * count, (3, 3), padding='same', name='block8_conv_bbox_0')(x), 'block8_bbox')
    block8_conv_class = reshaped_cls(Conv2D(count, (3, 3), padding='same', name='block8_conv_class_0')(x), 'block8_class')
    # Block 9
    block9_bbox = reshaped(Conv2D(4 * count, (3, 3), padding='valid', name='block9_conv_bbox_0')(x), 'block9_bbox')
    block9_conv_class = reshaped_cls(Conv2D(count, (3, 3), padding='valid', name='block9_conv_class_0')(x), 'block9_class')

    # Create model.
    inputs = img_input
    bbox_out = [block4_bbox, block5_bbox, block6_bbox, block7_bbox, block8_bbox, block9_bbox]
    cls_out = [block4_conv_class, block5_conv_class, block6_conv_class, block7_conv_class, block8_conv_class,
               block9_conv_class]
    model = Model(inputs, bbox_out + cls_out, name='ssd')
    return model, bbox_out, cls_out


def copy_weight_and_freeze(ssd, vgg16, freeze_classes):
    for layer in vgg16.layers:
        # print layer.__class__.__name__
        if not layer.__class__.__name__ == 'InputLayer':
            name = layer.name
            # print name
            ssd_layer = ssd.get_layer(name)
            ssd_layer.set_weights(layer.get_weights())
            ssd_layer.trainable = False

    if freeze_classes:
        for layer in ssd.layers:
            if layer.name.endswith('class_0'):
                print 'froze', layer.name
                layer.trainable = False


class SSDModel:
    def __init__(self ,freeze_classes=False):
        self.vgg16 = keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet', input_tensor=None, input_shape=(300, 300, 3))

        self.model, self.bbox_out, self.cls_out = get_ssd_model_description(6)
        copy_weight_and_freeze(self.model, self.vgg16, freeze_classes)

        self.shapes = [(x[1], x[2]) for x in map(lambda y: y._keras_shape, self.bbox_out)]

        get_scale = lambda shape: 2 * math.sqrt((1.0 / shape[0]) * (1.0 / shape[1]))
        self.scales = np.append([get_scale(shape) for shape in self.shapes], [1.0])

        self.scales = np.append(np.linspace(0.2, 0.9, len(self.shapes)), [1.0])
        self.ratios = [1, 2, 3, 0.5, 1.0 / 3]

        self.bbox_names = [bbox.name for bbox in self.model.layers if bbox.name.endswith('bbox')]
        self.cls_names = [bbox.name for bbox in self.model.layers if bbox.name.endswith('class')]

    def describe_params(self):
        print 'Shapes:', self.shapes
        print 'Scales:', self.scales
        print 'Ratios:', self.ratios
