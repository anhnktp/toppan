import numpy as np
import tensorflow as tf
from abc import abstractmethod
from helpers.settings import *
from helpers.bbox_utils import min_box_iou
from .DetectorBase import DetectorBase


class Detector(DetectorBase):
    """
    Using SSD version
    """

    def __init__(self, model_path, list_gpu, score_threshold, nms_threshold):
        super(Detector, self).__init__()

        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()

        # configuration for possible GPU use
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = list_gpu

        # the TensorFlow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            # Define input and output tensors (i.e. data) for the object detection classifier
            # Input tensor is the image
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Output tensors are the detection boxes, scores, and classes
            # Each box represents a part of the image where a particular object was detected
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represents level of confidence for each of the objects.
            # The score is shown on the result image, together with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            # Number of objects detected
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def box_normal_to_pixel(self, box, dim):
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
        return np.array(box_pixel)

    @abstractmethod
    def check_condition_bbox(self, dim, w, h):
        """
        @Get output of Detection models
        @return:
            cam_id: the id of camera
            frame_id: the id of frame image
            frame_data: the source of frame image
            boxes: array of objects
                Detection: object
                    box_geometry: [coordinates]
                    confidence: the result of Detection models
        """
        return True

    def getOutput(self):
        """
            boxes: array of objects
                    Detection: o
                        box_geometry: [coordinates]
                        confidence: the result of Detection models
                        labelbject: the label of object
        :param models:
        :param image:
        :param threshold:
        :return:
        """

        _img = self.frame['data']
        person_dets = []
        basket_dets = []
        with self.detection_graph.as_default():
            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            image_expanded = np.expand_dims(_img, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num_detections) = self.sess.run([self.detection_boxes,
                                                                      self.detection_scores,
                                                                      self.detection_classes,
                                                                      self.num_detections],
                                                                     feed_dict={self.image_tensor: image_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            # 1: person class
            # 2: basket class
            idx_vec = [i for i, v in enumerate(classes) if (((v == 1) or (v==2)) and (scores[i] > self._score_threshold))]
            if len(idx_vec) == 0:
                return np.asarray(person_dets), np.asarray(basket_dets)

            dim = _img.shape[0:2]
            for idx in idx_vec:
                # box = ymin,xmin,ymax,xmax
                box = self.box_normal_to_pixel(boxes[idx], dim)
                xmin = box[1] + self.roi_x1y1[0]
                ymin = box[0] + self.roi_x1y1[1]
                xmax = box[3] + self.roi_x1y1[0]
                ymax = box[2] + self.roi_x1y1[1]
                # Only keep person bbox has Square_bbox < 2/3*Square_image and Square_bbox > 100 pixel
                if (classes[idx] == 1) and self.check_condition_bbox(dim, xmax - xmin, ymax - ymin):
                    person_dets.append([xmin, ymin, xmax, ymax, scores[idx]])
                if (classes[idx] == 2):
                    basket_dets.append([xmin, ymin, xmax, ymax, scores[idx]])

            self.post_process(person_dets)
            self.post_process(basket_dets)

            return np.asarray(person_dets), np.asarray(basket_dets)

    def post_process(self, dets):
        del_index = []
        for i in range(len(dets) - 1):
            for j in range(i + 1, len(dets)):
                iou_ij = min_box_iou(dets[i], dets[j])
                if iou_ij > self._nms_threshold:
                    if (dets[i][-1] >= dets[j][-1]):
                        del_index.append(j)
                    else:
                        del_index.append(i)
                        continue
        del_index = np.unique(del_index)
        for i in reversed(del_index):
            dets.pop(i)

class PersonDetector(Detector):
    """
    Using SSD Resnet-50 version
    """

    def __init__(self):
        super(PersonDetector, self).__init__(os.getenv('SSD_MODEL_PATH'), os.getenv('CAM_360_GPU'),
                                             float(os.getenv('SCORE_THRESHOLD')), float(os.getenv('NMS_THRESHOLD')))

    def check_condition_bbox(self, dim, w, h):
        if (w * h < 0.3 * dim[0] * dim[1]) and (w * h > 200):
            return True
        return False

