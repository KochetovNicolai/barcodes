import keras
import os
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Reshape, Activation
from keras.applications.vgg16 import VGG16
import math
import numpy as np


def get_ssd_model_description(shape=(None, None, 3)):

    def activation(layer, name, activation):
        print name, layer._keras_shape
        return Activation(activation, name=name)(layer)

    img_input = Input(shape=shape, name='input')

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
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Note: vgg16 has 5 Blocks. Block 6 adds trainable params.

    # Block 6
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1), name='block6_pool')(x)

    bbox_out = []
    cls_out = []

    for coodr_name in ['dleft', 'dtop', 'dright', 'dbottom']:
        pref = coodr_name + '_'
        name = pref + 'block4_bbox'
        bbox_out.append(activation(Conv2D(1, (3, 3), padding='same', name=pref + 'block4_conv_bbox_0')(x), name, 'sigmoid'))

    cls_out.append(activation(Conv2D(1, (3, 3), padding='same', name='block4_conv_class_0')(x), 'block4_class', 'sigmoid'))

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
    def __init__(self ,freeze_classes=False, input_shape=(None, None, 3), do_not_load=False):

        if not do_not_load:
            self.vgg16 = keras.applications.vgg16.VGG16(
                include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape)

            self.model, self.bbox_out, self.cls_out = get_ssd_model_description(input_shape)
            copy_weight_and_freeze(self.model, self.vgg16, freeze_classes)

            self.bbox_names = [bbox.name for bbox in self.model.layers if bbox.name.endswith('bbox')]
            self.cls_names = [bbox.name for bbox in self.model.layers if bbox.name.endswith('class')]

        self.num_poolings = 6
        # Here we assume that it's possible to detect barcode from 64x64 image.
        self.window_size = 1 << self.num_poolings

        self.max_ratio = 3.0

    def describe_params(self):
        print 'Window size:', self.window_size
        print 'Max ratio:', self.max_ratio
        print 'Classes layers:'
        print '\n'.join(map(lambda x: '\t' + x, self.cls_names))
        print 'Bbox layers:'
        print '\n'.join(map(lambda x: '\t' + x, self.bbox_names))
