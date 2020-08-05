import torch
from .DetectorBase import DetectorBase
from .yolov5.models.yolo import Model
from .yolov5.models.yolo_new import Model as newModel
from .yolov5.utils.datasets import letterbox
from .yolov5.utils.general import non_max_suppression, scale_coords
from .yolov5.utils.torch_utils import select_device
from helpers.settings import *
from .sort import *

tracker = Sort()

class HandDetector(DetectorBase):
    """
        Using YOLOv5 version
    """
    def __init__(self, device, cfg_path, ckpt_path, augment=False, img_size=640):
        super(DetectorBase, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.device = select_device(device)
        self.augment = augment
        self.model = newModel(cfg_path)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model_state_dict'])
        self.model.fuse()
        self.model.to(self.device).eval()
        self.conf_thres = float(os.getenv('HAND_SCORE_THRESHOLD_YOLOV5'))
        self.iou_thres = float(os.getenv('HAND_NMS_THRESHOLD_YOLOV5'))
        self.img_size = img_size

    @torch.no_grad()
    def getOutput(self, current_time):
        img = self.frame['data']
        # Padded resize
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        img = img.unsqueeze(0)
        pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=False)
        dets = []
        det_trk = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (self.roi_x2y2[1], self.roi_x2y2[0])).round()
                for *xyxy, conf, cat in det:
                    x0, y0, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    xc = (x0 + x2) / 2.0
                    yc = (y0 + y2) / 2.0
                    x0, y0, x2, y2 = int(x0), int(y0), int(x2), int(y2)
                    xc, yc = int(xc), int(yc)
                    #dets.append([x0, y0, x2, y2, round(conf.item(), 2), (xc, yc)])
                    dets.append([x0, y0, x2, y2, round(conf.item(), 2)])
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


class PersonFaceDetector(DetectorBase):
    """
        Using YOLOv5 version
    """

    def __init__(self, device, cfg_path, ckpt_path, augment=False, img_size=(512, 512)):
        super(DetectorBase, self).__init__()
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        self.device = select_device(device)
        self.augment = augment
        self.model = Model(cfg_path)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        # self.model = torch.load(ckpt_path, map_location=self.device)['model'].float()
        # torch.save(self.model.state_dict(), 'signage_person_face.pt')
        self.model.fuse()
        self.model.to(self.device).eval()
        self.conf_thres = float(os.getenv('SCORE_THRESHOLD'))
        self.iou_thres = float(os.getenv('NMS_THRESHOLD'))
        self.img_size = img_size

    def getOutput(self):
        img = self.frame['data']
        # Padded resize
        img = letterbox(img, new_shape=self.img_size, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        img = img.unsqueeze(0)
        # save the memory when inference
        with torch.no_grad():
            pred = self.model(img, augment=self.augment)[0]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        person_dets = []
        face_dets = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (self.roi_x2y2[1], self.roi_x2y2[0])).round()
                dets = det.cpu().detach().numpy()
                person_dets = dets[dets[:, -1] == 0][:, 0:-1]
                face_dets = dets[dets[:, -1] == 1][:, 0:-1]
                return person_dets, face_dets
        return np.asarray(person_dets), np.asarray(face_dets)


class PersonBasketDetector(DetectorBase):
    """
        Using YOLOv5 version
    """

    def __init__(self, device, cfg_path, ckpt_path, augment=False, img_size=(512, 512)):
        super(DetectorBase, self).__init__()
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        self.device = select_device(device)
        self.augment = augment
        self.model = newModel(cfg_path)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.fuse()
        self.model.to(self.device).eval()
        self.conf_thres = float(os.getenv('SCORE_THRESHOLD'))
        self.iou_thres = float(os.getenv('NMS_THRESHOLD'))
        self.img_size = img_size


    def getOutput(self):
        img = self.frame['data']
        # Padded resize
        img = letterbox(img, new_shape=self.img_size, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        img = img.unsqueeze(0)
        # save the memory when inference
        with torch.no_grad():
            pred = self.model(img, augment=self.augment)[0]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        person_dets = []
        basket_dets = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (self.roi_x2y2[1], self.roi_x2y2[0])).round()
                dets = det.cpu().detach().numpy()
                person_dets = dets[dets[:, -1] == 0][:, 0:-1]
                basket_dets = dets[dets[:, -1] == 2][:, 0:-1]
                return person_dets, basket_dets
        return np.asarray(person_dets), np.asarray(basket_dets)