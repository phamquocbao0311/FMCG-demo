import numpy as np
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, \
    resize_image

import constants
from .base import BaseModel
from utils.files import ObjectDetectH5File

Threshold = 0.5
IOU = 0.3


class ObjectDetectModel(BaseModel):
    def __init__(self, file_path):
        load_file = ObjectDetectH5File(file_path)
        super(ObjectDetectModel, self).__init__(load_file)

    def predict(self, image):
        return self.model.predict_on_batch(np.expand_dims(image, axis=0))


class Detecting:
    def __init__(self, object_detect: ObjectDetectModel):
        self.object_detect = object_detect

    def predict_bb(self, name):
        image, scale = self.preprocessing_image(name)
        boxes, scores, labels = self.object_detect.predict(image)
        boxes = boxes / scale
        bb = BoundingBox.filter_bb(boxes, scores)

        return np.array(bb)

    def preprocessing_image(self, name):
        image = read_image_bgr(name)
        image = preprocess_image(image)
        return resize_image(image)


class BoundingBox:

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    @classmethod
    def filter_bb(cls, boxes, scores):
        bb = []

        # filter bounding box with IOU
        for i in range(len(boxes[0])):
            if scores[0][i] < Threshold:
                break
            for j in range(i - 1, -1, -1):

                area_iou = cls.bb_intersection_over_union(boxes[0][i],
                                                          boxes[0][j])
                if area_iou > IOU:
                    break
            else:
                bb.append(boxes[0][i])
        return bb


def get_detecting_model():
    object_detect = ObjectDetectModel(constants.OBJECT_DETECTION_MODEL_PATH)
    return Detecting(object_detect)
