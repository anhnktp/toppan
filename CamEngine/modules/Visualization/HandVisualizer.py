import cv2

class HandVisualizer:
    def __init__(self, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA):
        self.color = color
        self.thickness = thickness
        self.lineType = lineType

    def draw_boxes_2cams(self, dets_per_2_frames, frames):
        if len(dets_per_2_frames) > 0:
            for dets_per_frame, frame in zip(dets_per_2_frames, frames):
                if len(dets_per_frame) > 0:
                    for det in dets_per_frame:  # det = [x0, y0 , x1, y1, score]
                        det = det[:4]
                        # det = list(map(int, det))
                        p0 = (det[0], det[1])
                        p1 = (det[2], det[3])
                        cv2.rectangle(frame, p0, p1, self.color, self.thickness, self.lineType)
                        
    def draw_boxes(self, dets, frame):
        if len(dets) > 0:
            for det in dets:  # det = [x0, y0 , x1, y1, id, velocity, vx, vy, time, (xc, yc)]
                idx = str(int(det[4]))
                center = det[-1]
                det = det[:4]
                # det = list(map(int, det))
                p0 = (int(det[0]), int(det[1]))
                p1 = (int(det[2]), int(det[3]))
                cv2.rectangle(frame, p0, p1, self.color, self.thickness, self.lineType)
                cv2.circle(frame, center, 1, (255,0,0), 5)
                xc = int(0.5*(det[0]+det[2]))
                yc = int(0.5*(det[1]+det[3]))
                cv2.circle(frame, (xc, yc) , 1, (0, 255, 255), 5)
                cv2.putText(frame, idx, (int(det[0]), int(det[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)