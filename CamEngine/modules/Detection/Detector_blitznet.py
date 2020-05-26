import tensorflow as tf
from PIL import Image
import cv2
from .blitznet.config import config as net_config
from .blitznet.detector import Detector
from .blitznet.resnet import ResNet
from .blitznet.sort import *
from .DetectorBase import DetectorBase
from helpers.settings import *

class Loader():

    def __init__(self, cls_path):
        cats = self.load_classes(cls_path)
        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]

    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return ['__background__'] + list(filter(None, names)) + ['non-person']
        # filter removes empty strings (such as last line)

class PersonDetector(DetectorBase):

    def __init__(self, device, cls_path, ckpt_path, NO_SEG_GT=False):
        super(PersonDetector, self).__init__()
        self.net = ResNet(config=net_config, depth=50, training=False)
        self.loader = Loader(cls_path)
        self.model = ckpt_path
        self.no_seg_gt = NO_SEG_GT
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = device
        self.sess = tf.Session(config=config)
        self.detector = Detector(self.sess, self.net, self.loader, net_config, no_gt=self.no_seg_gt,
                                 CONF_THRESH=float(os.getenv('SCORE_THRESHOLD')), NMS_THRESH=float(os.getenv('NMS_THRESHOLD')))
        self.detector.restore_from_ckpt(self.model)

    def getOutput(self):
        img = self.frame['data']
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_img = np.array(pil_img) / 255.0
        pil_img = pil_img.astype(np.float32)
        h, w = pil_img.shape[:2]
        result = self.detector.feed_forward(img=pil_img, name=None, w=w, h=h, draw=False,
                                            seg_gt=None, gt_bboxes=None, gt_cats=None)
        bboxes = result[0]
        scores = result[1]
        cats = result[2]
        person_dets = []
        basket_dets = []
        for i in range(len(bboxes)):
            x_min = bboxes[i][0] + self.roi_x1y1[0]
            y_min = bboxes[i][1] + self.roi_x1y1[1]
            x_max = x_min + bboxes[i][2]
            y_max = y_min + bboxes[i][3]
            if int(cats[i]) == 1:
                person_dets.append([x_min, y_min, x_max, y_max, scores[i]])
            elif int(cats[i] == 3):
                basket_dets.append([x_min, y_min, x_max, y_max, scores[i]])

        return np.asarray(person_dets), np.asarray(basket_dets)
