import keras
import os
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Reshape, Activation
from keras.applications.vgg16 import VGG16
import math
import numpy as np


def get_ssd_model_description(ratios_names):
    def reshaped(layer, name):
        print layer._keras_shape
        return Activation('tanh', name=name)(layer)

    def reshaped_cls(layer, name):
        print layer._keras_shape
        return Activation('sigmoid', name=name)(layer)

    img_input = Input(shape=(None, None, 3), name='input')

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

    bbox_out = []
    cls_out = []
    for ratio_name in ratios_names:
        pref = ratio_name + '_'
        for coodr_name in ['dy', 'dx', 'sh', 'sw']:
            pref_pref = pref + coodr_name + '_'
            name = pref_pref + 'block4_bbox'
            bbox_out.append(reshaped(Conv2D(1, (3, 3), padding='same', name=pref_pref + 'block4_conv_bbox_0')(x), name))
        name = ratio_name + '_' + 'block4_class'
        cls_out.append(reshaped_cls(Conv2D(1, (3, 3), padding='same', name=pref + 'block4_conv_class_0')(x), name))

    # Create model.
    inputs = img_input
    model = Model(inputs, bbox_out + cls_out, name='ssd')
    return model, bbox_out, cls_out


def copy_weight_and_freeze(ssd, vgg16, freeze_classes):
    for layer in vgg16.layers:
        # print layer.__class__.__name__
        if not layer.__class__.__name__ == 'InputLayer':
            name = layer.name
            # print name
            ssd_layer = ssd.get_layer(name)
            if ssd_layer is None:
                print 'skip weight for:', name
                continue
            ssd_layer.set_weights(layer.get_weights())
            ssd_layer.trainable = False

    if freeze_classes:
        for layer in ssd.layers:
            if layer.name.endswith('class_0'):
                print 'froze', layer.name
                layer.trainable = False


class SSDModel:
    def __init__(self ,freeze_classes=False):

        self.ratios = [1, 2, 3, 0.5, 1.0 / 3]
        self.ratios_names = ["1x1", "2x1", "3x1", "1x2", "1x3"]

        self.vgg16 = keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet', input_tensor=None, input_shape=(300, 300, 3))

        self.model, self.bbox_out, self.cls_out = get_ssd_model_description(self.ratios_names)
        copy_weight_and_freeze(self.model, self.vgg16, freeze_classes)

        self.bbox_names = [bbox.name for bbox in self.model.layers if bbox.name.endswith('bbox')]
        self.cls_names = [bbox.name for bbox in self.model.layers if bbox.name.endswith('class')]

        scale = 1.0 / 16

        self.divisors = [1, 2, 4, 8, 16]
        self.shapes = [(int(37 / divisor), int(37 / divisor)) for divisor in self.divisors]
        self.scales = [scale * divisor for divisor in self.divisors]

    def describe_params(self):
        print 'Ratios:', self.ratios
        print 'Scales:', self.scales
        print 'Shapes:', self.shapes
