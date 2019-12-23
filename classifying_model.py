from keras.models import load_model, Model
from keras.layers import Layer
from keras import regularizers
from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import keras_efficientnets
n_classes = 143
n_cluster = 182

class Classifying:
    def __init__(self, Model_path, dir_cluster):
        self.classify = load_model(Model_path, custom_objects={'ArcFace': ArcFace})
        self.embedding = Model(self.classify.input[0], self.classify.layers[-3].output)
        self.cluster = np.load(dir_cluster)

        self.neigh = KNeighborsClassifier(n_neighbors=1)
        self.neigh.fit(self.cluster, [i for i in range(n_cluster)])

    def predict_class(self, image):
        image = np.array(image)/255.
        image = np.reshape(image,(1,80,80,3))
        feature = self.embedding.predict(image, verbose=1)
        feature /= np.linalg.norm(feature, axis=1, keepdims=True)
        dist, ind = self.neigh.kneighbors(feature, 1, True)
        return dist, ind

class ArcFace(Layer):
    def __init__(self, n_classes=n_classes, s=15., m=0.35, regularizer=None, **kwargs):
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
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)