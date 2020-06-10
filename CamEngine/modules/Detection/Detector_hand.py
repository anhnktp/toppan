# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from .DetectorBase import DetectorBase
from .sort import *
import numpy as np
import cv2
import torch
import time
from .utils.nms import nms

tracker = Sort()

class HandDetector(DetectorBase):
    def __init__(self, config_path, model_path, device, score_threshold, nms_thresh, box_area_thresh):
        super(HandDetector, self).__init__()
        self.score_threshold = float(score_threshold)
        self.nms_thresh = float(nms_thresh)
        self.box_area_thresh = float(box_area_thresh) 

        # Load config
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold  # set threshold for this model
        self.cfg.MODEL.DEVICE = f'cuda:{device}'
        self.cfg.MODEL.WEIGHTS = model_path
        
        # Create predictor
        self.predictor = DefaultPredictor(self.cfg)
    
    def getOutput(self, current_time):
        self.img = self.frame['data']
        # Make prediction for an image

        prediction = self.predictor(self.img)
        
        det_trk = []

        if len(prediction) > 0:
            outputs = prediction["instances"].to('cpu')
            boxes = outputs.pred_boxes
            scores = outputs.scores

            # Apply NMS
            keep_idx = self.nms(boxes, self.nms_thresh)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]

            dets = []
            for i in range(len(boxes)):
                score = scores[i].item()
                if score < self.score_threshold : continue
                if boxes[i].area() < self.box_area_thresh: continue
                
                x0 = boxes[i].tensor[0:1, 0:1].item() + self.roi_x1y1[0]
                y0 = boxes[i].tensor[0:1, 1:2].item() + self.roi_x1y1[1]
                x1 = boxes[i].tensor[0:1, 2:3].item() + self.roi_x1y1[0]
                y1 = boxes[i].tensor[0:1, 3:4].item() + self.roi_x1y1[1]
                xc = (x0 + x1) / 2.0
                yc = (y0 + y1) / 2.0
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                xc, yc = int(xc), int(yc)
                #dets.append([x0, y0, x1, y1, round(score, 2), (xc, yc)])
                dets.append([x0, y0, x1, y1, score])
            np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
            dets = np.asarray(dets)
            try:
                if len(dets) > 0:
                    tracks = tracker.update(dets)
                    for track in tracks:
                        track = track.tolist()
                        track.append(current_time)
                        xc = int(0.5*(track[0] + track[2]))
                        yc = int(0.5*(track[1] + track[3]))
                        track.append((xc,yc))
                        det_trk.append(track)
            except:
                print('unable to track')
        # det_trk format is [xmin, ymin, xmax, ymax, id, time, (xcenter, ycenter)]

        return det_trk

    
    def setFrames(self, frames_data):
        """
        Batch processing
        frames_data: list of images
        """
        self.__frames_data = frames_data

    @property
    def frames(self):
        """
        Batch processing
        """
        return {"data": self.__frames_data}

    def getOutputs(self):
        """
        Batch processing
        """
        self.imgs = self.frames['data']
        # Make prediction for a pair of images
        predictions = self.predictor(self.imgs)

        # Process prediction in each image
        dets = []   # store handboxes in both images
        for prediction in predictions:
            if len(prediction) > 0:
                outputs = prediction["instances"].to('cpu')
                boxes = outputs.pred_boxes
                scores = outputs.scores

                # Apply NMS
                keep_idx = self.nms(boxes, self.nms_thresh)
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]

                dets_per_img = []
                for i in range(len(boxes)):
                    score = scores[i].item()
                    if score < self.score_threshold : continue
                    if boxes[i].area() < self.box_area_thresh: continue
                    
                    x0 = boxes[i].tensor[0:1, 0:1].item() + self.roi_x1y1[0]
                    y0 = boxes[i].tensor[0:1, 1:2].item() + self.roi_x1y1[1]
                    x1 = boxes[i].tensor[0:1, 2:3].item() + self.roi_x1y1[0]
                    y1 = boxes[i].tensor[0:1, 3:4].item() + self.roi_x1y1[1]
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                    dets_per_img.append([x0, y0, x1, y1, score])
                tracker = Sort()
                tracks = tracker.update(dets)
                dets.append(tracks)
        return dets
    
    
    @staticmethod
    def nms(boxes, nms_thresh=0.5):
        bboxes = boxes.tensor.numpy()
        keep_idx = nms(bboxes, nms_thresh)
        return keep_idx


class BatchPredictor(DefaultPredictor):
    """
    Support batch inference
    """

    def __init__(self, cfg):
        super(BatchPredictor, self).__init__(cfg)

    def __call__(self, original_images):
        """
        Args:
            original_images: batch if images of shape (H, W, C) (in BGR order).
            each image: (np.ndarray)
        Returns:
            predictions (dict):
                the output of the model for batch of images
                See :doc:`/tutorials/models` for details about the format.
        """
 
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # whether the model expects BGR inputs or RGB
            inputs = []
            for i, original_image in enumerate(original_images):
                if self.input_format == "RGB":
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.transform_gen.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input = {"image": image, "height": height, "width": width}
                inputs.append(input)

            predictions = self.model(inputs)
            return predictions
