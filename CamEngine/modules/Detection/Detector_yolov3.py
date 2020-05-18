from .DetectorBase import DetectorBase
from .yolov3.models import *  # set ONNX_EXPORT in models.py
from .yolov3.utils.datasets import *
from .yolov3.utils.utils import *
from helpers.settings import *


class PersonDetector(DetectorBase):
    """
        Using YOLOv3 version
    """
    def __init__(self, gpu_id, cfg_path, ckpt_path, cls_names, augment):
        super(DetectorBase, self).__init__()
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        self.device = torch_utils.select_device(gpu_id)
        self.augment = augment
        img_size = (512, 512)
        self.model = Darknet(cfg_path, img_size)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model'])
        self.model.to(self.device).eval()

        self.names = load_classes(cls_names)
        self.conf_thres = 0.65
        self.iou_thres = 0.3
        # self.model.fuse()

    def getOutput(self):
        img = self.frame['data']
        # Padded resize
        img = letterbox(img, new_shape=512)[0]
        # img = cv2.resize(img, (512, 512))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        img = img.unsqueeze(0)
        pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, multi_label=False, agnostic=False)
        dets = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (self.roi_x2y2[1], self.roi_x2y2[0])).round()
                dets = det.cpu().detach().numpy()[:, 0:-1]
                dets = dets[dets[:, -1] < 1]
                return dets
        return np.asarray(dets)

if __name__ == '__main__':
    cfg_path = '/home/anhvn/yolov3/cfg/yolov3-spp-1cls.cfg'
    model_path = '/home/anhvn/yolov3/weights/best.pt'
    cls_names = '/home/anhvn/yolov3/data/omni.names'
    vid_path = '/mnt/ssd2/Datasets/Fish_eye_dataset/PJ9/Toppan_15_04_20/12_3ppl/07_center_fisheye_2020_04_15_14_30_06.mp4'
    detector = PersonDetector(os.getenv('CAM_360_GPU'), cfg_path, ckpt_path=model_path, cls_names=cls_names, augment=False)
    cap = cv2.VideoCapture(vid_path)
    cnt = 0
    while cap.isOpened():
        _, img = cap.read()
        start_time = time.time()

        detector.setFrame(img, cnt)
        dets = detector.getOutput()
        print(dets)
        total_time = time.time() - start_time
        print('--------------- FPS: {} ----------------'.format(1./ total_time))

