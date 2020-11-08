import json
from abc import abstractmethod
import numpy as np
from keras_retinanet import models
from keras.models import load_model, Model
from keras.layers import Layer
from keras import regularizers
from keras import backend as K
import tensorflow as tf

import constants
from constants import NUMBER_CLASS


class BaseFile:

    def __init__(self, file_path):
        self.file_path = file_path

    @abstractmethod
    def load(self):
        raise NotImplemented


class NumpyFile(BaseFile):

    def __init__(self, file_path):
        super(NumpyFile, self).__init__(file_path)

    def load(self):
        return np.load(self.file_path)


class ClassifyH5File(BaseFile):
    def __init__(self, file_path):
        super(ClassifyH5File, self).__init__(file_path)

    def load(self):
        classify = load_model(self.file_path,
                              custom_objects={'ArcFace': ArcFace})
        return Model(classify.input[0], classify.layers[-3].output)


class ObjectDetectH5File(BaseFile):
    def __init__(self, file_path):
        super(ObjectDetectH5File, self).__init__(file_path)

    def load(self):
        model = models.load_model(self.file_path, backbone_name='resnet50')
        return models.convert_model(model)


class JsonFile(BaseFile):
    def __init__(self, file_path):
        super(JsonFile, self).__init__(file_path)

    def load(self):
        with open(self.file_path) as f:
            load_file = json.load(f)
        return load_file


class ArcFace(Layer):
    def __init__(self, n_classes=NUMBER_CLASS, s=15., m=0.35, regularizer=None,
                 **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        # print(input_shape)
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)

        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return None, self.n_classes
