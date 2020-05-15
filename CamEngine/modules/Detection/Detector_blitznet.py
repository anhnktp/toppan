from modules.blitznet.config import config as net_config
from modules.blitznet.detector import Detector
from modules.blitznet.resnet import ResNet
import tensorflow as tf
from PIL import Image
from modules.blitznet.env import *
from modules.blitznet.sort import *
from modules.Base.DetectorBase import DetectorBase
import cv2

VOC_CATS = ['__background__','person','non-person']

class Loader():
    def __init__(self):
        cats = VOC_CATS
        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]

class Bliznet_detector(DetectorBase):
    def __init__(self):
        super(Bliznet_detector, self).__init__()
        self.net = ResNet(config=net_config, depth=50, training=False)
        self.loader = Loader()
        self.model = MODEL_PATH
        self.no_seg_gt = NO_SEG_GT
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = GPU_IDS
        self.sess = tf.Session(config=config)
        self.detector = Detector(self.sess, self.net, self.loader, net_config, no_gt=self.no_seg_gt)
        self.detector.restore_from_ckpt(self.model)

    def getOutput(self):
        img = self.frame['data']
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_img = np.array(pil_img) / 255.0
        pil_img = pil_img.astype(np.float32)
        h, w = pil_img.shape[:2]
        result = self.detector.feed_forward(img=pil_img, name=None, w=w, h=h, draw=False,
                                            seg_gt=None, gt_bboxes=None, gt_cats=None)
        bbox = result[0]
        score = result[1]
        cats = result[2]
        dets = []
        for i in range(len(bbox)):
            if cats[i] > 1:
                continue
            else:
                x_min = bbox[i][0] + self.roi_x1y1[0]
                y_min = bbox[i][1] + self.roi_x1y1[1]
                x_max = x_min + bbox[i][2]
                y_max = y_min + bbox[i][3]
                dets.append([x_min, y_min, x_max, y_max, score[i]])
        dets = np.asarray(dets)
        return dets

def main():
    video = cv2.VideoCapture(VID_PATH)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = int(video.get(cv2.CAP_PROP_FPS))
    print('Loaded video has WIDTH: {} - HEIGHT: {} - FPS: {} !'.format(width, height, frames_per_second))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    width, height = 512, 512
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, frames_per_second, (width, height))
    import time
    detector = Bliznet_detector()
    mot_tracker = Sort()
    total_time = 0
    colours = np.random.randint(0, 256, size=(32, 3))
    for frame_index in range(num_frames):
        frame_index += 1
        start_time = time.time()
        ret, img = video.read()
        img = cv2.resize(img, (512, 512))
        dets = detector.detect(img)
        trackers = mot_tracker.update(dets)
        for d in trackers:
            tl = 2
            c1, c2 = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
            color = colours[int(d[4]) % 32].tolist()
            cv2.rectangle(img, c1, c2, color, thickness=tl)
            # Plot score
            thickness = max(tl - 1, 1)  # font thickness
            label = str(int(d[4]))
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=thickness)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [0, 0, 0], thickness=thickness,
                                lineType=cv2.LINE_AA)
        videoWriter.write(img)
        print('FPS: {:.2f} ___________Follow___________ Frame: {:d}'.format(1 / (time.time() - start_time),
                                                                                    frame_index + 1))
        cycle_time = time.time() - start_time
        total_time += cycle_time
    print("Total Time_systems took: %.3f for %d frames or %.1f FPS" % (
        total_time, num_frames, num_frames / total_time))
    print('Done')
if __name__ == '__main__':
    main()
