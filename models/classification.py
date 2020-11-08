import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import keras_efficientnets

import constants
from .base import BaseModel
from utils.files import ClassifyH5File, NumpyFile

n_cluster = 182


class ClassifyingImage(BaseModel):
    def __init__(self, model_path):
        load_file = ClassifyH5File(model_path)
        super(ClassifyingImage, self).__init__(load_file)

    def predict_class(self, image):
        image = np.array(image) / 255.
        image = np.reshape(image, (1, 80, 80, 3))
        feature = self.model.predict(image, verbose=1)
        feature /= np.linalg.norm(feature, axis=1, keepdims=True)
        dist, ind = self.neigh.kneighbors(feature, 1, True)
        return dist, ind

    def predict(self, image):
        return self.model.predict(image, verbose=1)


class ClassifyingEmbedding(BaseModel):

    def __init__(self, embedding_path):
        load_file = NumpyFile(embedding_path)
        super(ClassifyingEmbedding, self).__init__(load_file)
        self.neigh = KNeighborsClassifier(n_neighbors=1)
        self.neigh.fit(self.model, [i for i in range(n_cluster)])

    def predict(self, feature):
        return self.neigh.kneighbors(feature, 1, True)


class Classifying:

    def __init__(self, classifying_image: ClassifyingImage,
                 classifying_embedding: ClassifyingEmbedding):
        self.classifying_image = classifying_image
        self.classifying_embedding = classifying_embedding

    def predict_class(self, image):
        image = self.preprocessing_image(image)
        feature = self.classifying_image.predict(image)
        feature /= np.linalg.norm(feature, axis=1, keepdims=True)
        return self.classifying_embedding.predict(feature)

    def preprocessing_image(self, image):
        image = np.array(image) / 255.
        image = np.reshape(image, (1, 80, 80, 3))
        return image


def get_classifying_model():
    classifying_image = ClassifyingImage(constants.CLASSIFYING_MODEL_PATH)
    classifying_embedding = ClassifyingEmbedding(constants.CLUSTER_PATH)
    return Classifying(classifying_image, classifying_embedding)
