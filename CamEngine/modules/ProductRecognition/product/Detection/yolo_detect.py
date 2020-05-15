from .Base import DetectionBase
import logging
from .config.yolo_config import *
from .models.yolo2 import YOLO
from modules.product.Common.utils import cv2_PIL
import os

logging.basicConfig(format='[%(levelname)s|%(asctime)s] %(message)s',
                    datefmt='%Y%m%d %H:%M:%S',
                    level=logging.INFO)


class Detection(DetectionBase):
    """
    Using YOLO version
    """
    def __init__(self):

        super(Detection, self).__init__()
        self.weight_file = os.path.join(os.path.dirname(__file__), 'weights', WEIGHTS)
        self.class_file = os.path.join(os.path.dirname(__file__), 'models', 'model_data', CLASSES_PATH)
        self.anchor_file = os.path.join(os.path.dirname(__file__), 'models', 'model_data', ANCHORS_PATH)
        self.model = YOLO( self.weight_file,classes_path=self.class_file, anchors_path=self.anchor_file)

    def getOutput(self, threshold=0.7):
        """
            boxes: array of objects
                    box_id: the id of bounding box
                    detection: object
                        box_geometry: [coordinates]
                        confidence: the result of detection model
                        label: the label of object
        :param model:
        :param image:
        :param threshold:
        :return:
        """
        boxes = []
        results = []
        _img = self.image['src']
        # _img = apply_brightness_contrast(_img, brightness = 0, contrast = 0)
        _img = cv2_PIL(_img[100:,:])


        if _img.mode != "RGB":
            _img = self.image.convert("RGB")
        try:
            results = self.model.detect(_img)

            # logging.info("Detection Image. Time: {}".format(time.time() - start_time))
        except Exception as ex:
            logging.error("Detect Image error: {}".format(ex))
        result = {}
        for i, re in enumerate(results):
            im_tag = re[0].split(' ')[0]
            box = {}

            if im_tag == 'satudora_product':
                # print(re)
                if float(re[0].split(' ')[1]) > threshold:
                    box['box_id'] = i
                    box['detection'] = {}
                    box['detection']['coordinates'] = [int(x) + 100 * ((i) % 2) for i, x in enumerate(re[1:])]
                    # box['detection']['coordinates'] = [int(x) for i, x in enumerate(re[1:])]
                    box['detection']['confidence'] = float(re[0].split(' ')[1])
                    box['detection']['label'] = im_tag
                    boxes.append(box)
        result['boxes'] = boxes
        return result

