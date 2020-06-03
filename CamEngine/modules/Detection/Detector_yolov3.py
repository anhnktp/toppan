from .DetectorBase import DetectorBase
from .yolov3.models import *  # set ONNX_EXPORT in models.py
from .yolov3.utils.datasets import *
from .yolov3.utils.utils import *
from helpers.settings import *


class PersonDetector(DetectorBase):
    """
        Using YOLOv3 version
    """
    def __init__(self, device, cfg_path, ckpt_path, cls_names, augment=False, img_size=(512, 512)):
        super(DetectorBase, self).__init__()
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        self.device = torch_utils.select_device(device)
        self.augment = augment
        self.model = Darknet(cfg_path, img_size)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model'])
        self.model.to(self.device).eval()

        self.names = load_classes(cls_names)
        self.conf_thres = float(os.getenv('SCORE_THRESHOLD'))
        self.iou_thres = float(os.getenv('NMS_THRESHOLD'))
        # self.model.fuse()

    def getOutput(self):
        img = self.frame['data']
        # Padded resize
        img = letterbox(img, new_shape=512, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        img = img.unsqueeze(0)
        pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, multi_label=False, agnostic=False)
        person_dets = []
        basket_dets = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (self.roi_x2y2[1], self.roi_x2y2[0])).round()
                dets = det.cpu().detach().numpy()
                person_dets = dets[dets[:, -1] == 0][:, 0:-1]
                basket_dets = dets[dets[:, -1] == 1][:, 0:-1]
                return person_dets, basket_dets

        return np.asarray(person_dets), np.asarray(basket_dets)