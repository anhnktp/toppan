# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from .DetectorBase import DetectorBase
import numpy as np


class Detectron(DetectorBase):

    def __init__(self, device, config_path, model_path, score_threshold=0.8):
        super(Detectron, self).__init__()
        # Load config
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold  # set threshold for this model
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.WEIGHTS = model_path
        self.score_threshold = score_threshold
        # Create predictor
        self.predictor = DefaultPredictor(self.cfg)

    def getOutput(self):
        self.img = self.frame['data']
        # Make prediction
        outputs = self.predictor(self.img)
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        dets = []
        for i in range(len(boxes)):
            score = scores[i].item()
            if score < self.score_threshold : continue
            x0 = boxes[i].tensor[0:1, 0:1].item() + self.roi_x1y1[0]
            y0 = boxes[i].tensor[0:1, 1:2].item() + self.roi_x1y1[1]
            x1 = boxes[i].tensor[0:1, 2:3].item() + self.roi_x1y1[0]
            y1 = boxes[i].tensor[0:1, 3:4].item() + self.roi_x1y1[1]
            dets.append([x0, y0 , x1, y1, score])
        dets = np.array(dets)
        return dets

