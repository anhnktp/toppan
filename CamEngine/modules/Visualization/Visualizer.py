import numpy as np
from collections import deque
import cv2

class Visualizer(object):

    def __init__(self, TRAJECT_QUEUE_SIZE=64):
        super(Visualizer, self).__init__()
        # Create color table
        np.random.seed(10)
        self.colours = np.random.randint(0, 256, size=(16, 3))
        # Create trajectories point list
        # TRAJECT_QUEUE_SIZE buffer is the maximum size of plot deque
        self.pts = deque(maxlen=TRAJECT_QUEUE_SIZE)

    def draw(self, img, basket_dets, trackers, event_detector):
        plot_bbox(img, basket_dets, colours=self.colours, show_label=False)
        plot_bbox(img, trackers, colours=self.colours, show_label=True)
        plot_tracjectories(img, pts=self.pts, trackers=trackers, colours=self.colours)

        if len(event_detector.localIDs_entered) > 0:
            cv2.putText(img, 'ENTER:{}'.format(", ".join(map(str, event_detector.localIDs_entered))), (2, 55), 0, fontScale=0.6,
                        color=(0, 0, 255), thickness=2)
        if len(event_detector.localIDs_exited) > 0:
            cv2.putText(img, 'EXIT:{}'.format(", ".join(map(str, event_detector.localIDs_exited))), (2, 75), 0, fontScale=0.6,
                        color=(0, 0, 255), thickness=2)
        if len(event_detector.localIDs_A) > 0:
            cv2.putText(img, 'A:{}'.format(", ".join(map(str, event_detector.localIDs_A))), (2, 95), 0, fontScale=0.6,
                        color=(0, 0, 255), thickness=2)
        if len(event_detector.localIDs_B) > 0:
            cv2.putText(img, 'B:{}'.format(", ".join(map(str, event_detector.localIDs_B))), (2, 115), 0, fontScale=0.6,
                        color=(0, 0, 255), thickness=2)

    def draw_signage(self, img, faces, trackers, event_detector):
        plot_bbox(img, faces, colours=self.colours, show_label=False)
        plot_bbox(img, trackers, colours=self.colours, show_label=True)
        plot_tracjectories(img, pts=self.pts, trackers=trackers, colours=self.colours)

    def show(self, img, title='Demo'):
        cv2.imshow(title, img)
        key= cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            quit()        
        if key & 0xFF == ord('s'): # stop
            cv2.waitKey(0)


def plot_bbox(img, bboxes, colours, show_label=True):
    for d in bboxes:
        if d[-1] < 1: continue
        color = colours[int(d[-1]) % 16].tolist()
        tl = round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
        c1, c2 = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        # Plot score
        if not show_label: continue
        tf = max(tl - 1, 1)  # font thickness
        label = '%d' % int(d[-1])     # local_id
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [0, 0, 0], thickness=tf,
                    lineType=cv2.LINE_AA)

def plot_tracjectories(img, pts, trackers, colours):
    centers = []
    for d in trackers:
        if d[-1] < 0: continue
        centers.append((int((d[0] + d[2]) / 2), int((d[1] + d[3]) / 2), int(d[-1])))
    pts.appendleft(centers)

    # Plot trajectories
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        for j in range(0, len(pts[i - 1])):
            for k in range(0, len(pts[i])):
                if (pts[i - 1][j][2] == pts[i][k][2]) and (pts[i - 1][j][2] > 0):
                    color = colours[pts[i - 1][j][2] % 16].tolist()
                    cv2.line(img, pts[i - 1][j][0:2], pts[i][k][0:2], color, thickness=2)
                    continue