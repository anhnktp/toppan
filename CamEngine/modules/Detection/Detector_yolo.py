import cv2
import numpy as np
import tensorflow as tf

from .yolo.utils.data_aug import letterbox_resize
from .yolo.utils.misc_utils import parse_anchors, read_class_names
from .yolo.utils.nms_utils import gpu_nms
from helpers.settings import *
from .DetectorBase import DetectorBase
from .yolo.yolov3 import yolov3


class PersonDetector(DetectorBase):
    """
    Using YOLOv3 version
    """

    def __init__(self):
        super(PersonDetector, self).__init__()
        self._input_width = int(os.getenv('INPUT_SIZE').split(',')[0])
        self._input_height = int(os.getenv('INPUT_SIZE').split(',')[1])
        self._anchors = parse_anchors(join(ROOT_DIR, os.getenv('ANCHORS_PATH')))
        self._classes = read_class_names(join(ROOT_DIR, os.getenv('CLASSES_PATH')))
        self._num_class = len(self._classes) or os.getenv('NUM_CLASS')
        self._letterbox_resize = int(os.getenv('LETTERBOX_RESIZE'))

        # configuration for possible GPU use
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        # the TensorFlow graph
        self.sess = tf.compat.v1.Session(config=config)
        self.input_data = tf.compat.v1.placeholder(tf.float32, [1, self._input_height, self._input_width, 3],
                                                   name='input_data')
        self.yolo_model = yolov3(self._num_class, self._anchors)
        with tf.compat.v1.variable_scope('yolov3'):
            pred_feature_maps = self.yolo_model.forward(self.input_data, False)
        pred_boxes, pred_confs, pred_probs = self.yolo_model.predict(pred_feature_maps)
        pred_scores = pred_confs * pred_probs
        self.boxes, self.scores, self.labels = gpu_nms(pred_boxes, pred_scores, self._num_class,
                                                       max_boxes=int(os.getenv('MAX_BOXES')),
                                                       score_thresh=float(os.getenv('SCORE_THRESHOLD')),
                                                       nms_thresh=float(os.getenv('NMS_THRESHOLD')))
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, join(ROOT_DIR, os.getenv('RESTORE_PATH')))

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
        if self._letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(_img, self._input_width, self._input_height)
        else:
            height_ori, width_ori = _img.shape[:2]
            img = cv2.resize(_img, (self._input_width, self._input_height))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        try:
            boxes_, scores_, labels_ = self.sess.run([self.boxes, self.scores, self.labels],
                                                     feed_dict={self.input_data: img})
            # rescale the coordinates to the original image
            if self._letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori / float(self._input_width))
                boxes_[:, [1, 3]] *= (height_ori / float(self._input_height))

            results = []
            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                if labels_[i] == 0: results.append([x0, y0, x1, y1, scores_[i]])

            return np.asarray(results)  # [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score], ..]
        except Exception as ex:
            engine_logger.error("Detect frameID {} - Error: {}".format(self.frame['id'], ex))
            return None
