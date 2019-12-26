import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

Threshold = 0.5
IOU = 0.3

class Model:
    def __init__(self, Model_path):
        self.model = models.load_model(Model_path, backbone_name='resnet50')
        self.model = models.convert_model(self.model)

    def predict_bb(self, name):
        image = read_image_bgr(name)
        image = preprocess_image(image)
        image,scale = resize_image(image)
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes = boxes/scale
        bb = []

        #filter bounding box with IOU
        for i in range(len(boxes[0])):
            if scores[0][i] < Threshold:
                break
            for j in range(i-1,-1,-1):
                if self.bb_intersection_over_union(boxes[0][i], boxes[0][j]) > IOU:
                    break
            else:
                bb.append(boxes[0][i])

        return np.array(bb)

    def bb_intersection_over_union(self, boxA, boxB):
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