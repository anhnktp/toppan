import numpy as np
from shapely.geometry import Point
from .TrackerBase import TrackerBase
from .utils.KalmanBox import KalmanBoxTracker
from .utils.associate_dets_trks import associate_detections_to_trackers
from .utils.check_accompany import compare_2bboxes_area
from helpers.common_utils import calculate_duration
import os

def find_area(bbox, in_door_box, out_door_box, a_box, b_box):
    center_point = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    top_left_point = Point(bbox[0], bbox[1])
    bottom_left_point = Point(bbox[0], bbox[3])
    if out_door_box.contains(bottom_left_point) or out_door_box.contains(top_left_point):
    # if out_door_box.contains(bottom_left_point):
        return 'OUT_DOOR_AREA'
    if in_door_box.contains(bottom_left_point) or in_door_box.contains(top_left_point):
    # if in_door_box.contains(bottom_left_point):
        return 'IN_DOOR_AREA'
    if a_box.contains(center_point):
        return 'A_AREA'
    if b_box.contains(center_point):
        return 'B_AREA'
    return None

class Tracker(TrackerBase):
    """
       Using SORT Tracking
    """
    def __init__(self, max_age=20, min_hits=5, low_iou_threshold=0.25, min_dist_ppl=50, min_freq_ppl=15):
        """
        Sets key parameters for SORT
        """
        super(Tracker, self).__init__()
        self._max_age = max_age
        self._min_hits = min_hits
        self._low_iou_threshold = low_iou_threshold
        self._trackers = []
        self._min_dist_ppl = min_dist_ppl
        self._min_freq_ppl = min_freq_ppl

    def update(self, dets, baskets, in_door_box, out_door_box, a_box, b_box, none_box):

        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            Requires: this method must be called once for each frame even with empty detections.
            NOTE: The number of objects returned may differ from the number of detections provided.
            STRUCTURE of each return track: [xmin, ymin, xmax, ymax, bit_area, cur_time, basket_count, basket_time, local_id]
            bit_area: 1 = ENTER, 2 = EXIT, 3 = A_AREA, 4 = B_AREA, -1 = NOT IN SPECIAL AREA
            cur_time: current timestamp
            basket_count: number frames track has basket
            basket_time: if basket_count > BASKET_FREQ, basket_time is first time track has basket else first time create track
            local_id: ID of track
        """
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self._trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self._trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self._trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self._low_iou_threshold)
        res = np.zeros((len(dets), 9))
        # update matched trackers with assigned detections
        for d, t in matched:
            area = find_area(dets[d, 0:-1], in_door_box, out_door_box, a_box, b_box)
            if (area is not None) and (self._trackers[t].area != area):
                if (self._trackers[t].area == 'OUT_DOOR_AREA') and (area == 'IN_DOOR_AREA'): res[d, -5] = 1  # 1 = ENTER
                if (self._trackers[t].area == 'IN_DOOR_AREA') and (area == 'OUT_DOOR_AREA'): res[d, -5] = 2  # 2 = EXIT
                if (area == 'A_AREA'): res[d, -5] = 3  # 3 = A
                if (area == 'B_AREA'): res[d, -5] = 4  # 4 = B
                self._trackers[t].area = area
            else:
                res[d, -3] = -1  # -1 = None move to special area
            self._trackers[t].update(dets[d], self._min_hits)
            res[d, 0:4] = dets[d, 0:-1]
            res[d, -1] = self._trackers[t].id
            res[d, -2] = self._trackers[t].basket_time
            res[d, -3] = self._trackers[t].basket_count
            res[d, -4] = self._timestamp

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            center_point = Point((dets[i, 0] + dets[i, 2]) / 2, (dets[i, 1] + dets[i, 3]) / 2)
            if none_box.contains(center_point): continue
            trk = KalmanBoxTracker(dets[i], self._timestamp)
            self._trackers.append(trk)
            res[i, 0:4] = dets[i, 0:-1]
            res[i, -1] = trk.id
            res[i, -2] = trk.basket_time
            res[i, -3] = trk.basket_count
            res[i, -4] = self._timestamp
            bbox = dets[i, 0:-1]
            area = find_area(bbox, in_door_box, out_door_box, a_box, b_box)
            if (area is not None) and (trk.area != area):
                if (trk.area == 'OUT_DOOR_AREA') and (area == 'IN_DOOR_AREA'): res[i, -5] = 1  # 1 = ENTER
                if (trk.area == 'IN_DOOR_AREA') and (area == 'OUT_DOOR_AREA'): res[i, -5] = 2  # 2 = EXIT
                if (area == 'A_AREA'): res[i, -5] = 3  # 3 = A
                if (area == 'B_AREA'): res[i, -5] = 4  # 4 = B
                trk.area = area
            else:
                res[i, -3] = -1  # -1 = None move to special area

        # Update basket to existing tracks
        self.associate_basket2trackers(baskets)

        # Update accompany people to existing tracks
        self.count_accompany_ppl2trackers(res)

        # remove dead tracklet
        i = len(self._trackers)
        localIDs_end = []
        for trk in reversed(self._trackers):
            i -= 1
            if (trk.time_since_update > self._max_age):
                # last_state = trk.get_last_state(-self._max_age)
                ppl_accompany = np.asarray(list(trk.ppl_dist.values()))
                ppl_accompany = ppl_accompany[ppl_accompany > self._min_freq_ppl]
                localIDs_end.append([trk.id, trk.basket_count, int(trk.basket_time), len(ppl_accompany), int(self._timestamp)])
                self._trackers.pop(i)

        if (len(res) > 0):
            return res, localIDs_end
        return np.empty((0, 9)), localIDs_end

    def count_accompany_ppl2trackers(self, res):
        for i, trki in enumerate(res):
            for j, trkj in enumerate(res):
                if (j == i) or (trki[-1] < 1) or (trkj[-1] < 1): continue
                point_trki = np.array(((trki[0] + trki[2]) / 2, (trki[1] + trki[3]) / 2))
                point_trkj = np.array(((trkj[0] + trkj[2]) / 2, (trkj[1] + trkj[3]) / 2))
                dist = np.linalg.norm(point_trki - point_trkj)
                if dist > self._min_dist_ppl:
                    for trk in self._trackers:
                        if trk.id == int(trki[-1]):
                            trk.ppl_dist[int(trkj[-1])] = trk.ppl_dist.get(int(trkj[-1]), 0) + 1

    def associate_basket2trackers(self, baskets):
        # get locations from existing trackers.
        trks = np.zeros((len(self._trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self._trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self._trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(baskets, trks, low_iou_threshold=0.1)
        for t in matched:
            if self._trackers[t[1]].basket_count == 0:
                self._trackers[t[1]].basket_time = self._timestamp
            self._trackers[t[1]].basket_count += 1
            baskets[t[0], -1] = self._trackers[t[1]].id
        # for t in unmatched_dets:
        #     baskets[t, -1] = -1     # basket not assigned has id = -1

class Tracker1(TrackerBase):
    """
       Using SORT Tracking
    """
    def __init__(self, max_age, min_hits, low_iou_threshold=0.25):
        """
        Sets key parameters for SORT
        """
        super(Tracker1, self).__init__()
        self._max_age = max_age
        self._min_hits = min_hits
        self._low_iou_threshold = low_iou_threshold
        self._trackers = []

    def update(self, dets, basket_dets, in_door_box, out_door_box, a_box, b_box, none_box):

        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self._trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self._trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self._trackers[t].id]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self._trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self._low_iou_threshold)
        res = np.zeros((len(dets), 8))
        trks_start = []

        # update matched trackers with assigned detections
        to_del = []
        for t, trk in enumerate(self._trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                is_new_local_id = trk.update(dets[d[0]], self._min_hits)
                res[d[0], 0:-4] = dets[d[0], 0:-1]
                bbox = dets[d[0], 0:-1]
                res[d[0], -2] = trk.timestamp
                res[d[0], -1] = trk.id
                res[d[0], -4] = self._timestamp
                area = find_area(bbox, in_door_box, out_door_box, a_box, b_box)
                if (area is not None) and (trk.area != area):
                    if (trk.area == 'OUT_DOOR_AREA') and (area == 'IN_DOOR_AREA'): res[d[0], -3] = 1    # 1 = ENTER
                    if (trk.area == 'IN_DOOR_AREA') and (area == 'OUT_DOOR_AREA'):
                        res[d[0], -3] = 2    # 2 = EXIT
                        to_del.append(t)
                    if (area == 'A_AREA'): res[d[0], -3] = 3   # 3 = A
                    if (area == 'B_AREA'): res[d[0], -3] = 4   # 4 = B
                    trk.area = area
                else:
                    if (trk.area == 'OUT_DOOR_AREA'):
                        res[d[0], -3] = -2    # -1 = None move to special area
                    else:
                        res[d[0], -3] = -1

                if is_new_local_id:
                    trks_start.append(res[d[0]])

        for t in reversed(to_del):
            self._trackers.pop(t)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            center_point = Point((dets[i, 0] + dets[i, 2]) / 2, (dets[i, 1] + dets[i, 3]) / 2)
            if none_box.contains(center_point): continue
            trk = KalmanBoxTracker(dets[i], self._timestamp)
            self._trackers.append(trk)
            res[i, 0:-4] = dets[i, 0:-1]
            res[i, -2] = trk.timestamp
            res[i, -1] = trk.id
            res[i, -4] = self._timestamp
            bbox = dets[i, 0:-1]
            area = find_area(bbox, in_door_box, out_door_box, a_box, b_box)
            if (area is not None) and (trk.area != area):
                if (trk.area == 'OUT_DOOR_AREA') and (area == 'IN_DOOR_AREA'): res[i, -3] = 1  # 1 = ENTER
                if (trk.area == 'IN_DOOR_AREA') and (area == 'OUT_DOOR_AREA'): res[i, -3] = 2  # 2 = EXIT
                if (area == 'A_AREA'): res[i, -3] = 3  # 3 = A
                if (area == 'B_AREA'): res[i, -3] = 4  # 4 = B
                trk.area = area
            else:
                if (trk.area == 'OUT_DOOR_AREA'):
                    res[i, -3] = -2  # -1 = None move to special area
                else:
                    res[i, -3] = -1

        i = len(self._trackers)
        # remove dead tracklet
        localIDs_end = []
        for trk in reversed(self._trackers):
            i -= 1
            if (trk.time_since_update > self._max_age):
                # last_state = trk.get_last_state(-self._max_age)
                localIDs_end.append(trk.id)
                self._trackers.pop(i)
        if (len(res) > 0):
            return res, np.asarray(trks_start), localIDs_end
        return np.empty((0, 8)), np.asarray(trks_start), localIDs_end

class SignageTracker(TrackerBase):
    """
       Using SORT Tracking
    """
    def __init__(self, max_age=20, min_hits=5, low_iou_threshold=0.25, min_area_ratio=0.5, max_area_ratio=1.5, min_area_freq=10):
        """
        Sets key parameters for SORT
        """
        super(SignageTracker, self).__init__()
        self._max_age = max_age
        self._min_hits = min_hits
        self._low_iou_threshold = low_iou_threshold
        self._trackers = []
        self._min_area_ratio = min_area_ratio
        self._max_area_ratio = max_area_ratio
        self._min_area_freq = min_area_freq

    def update(self, dets, faces,headpose_Detector,frame):

        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            Requires: this method must be called once for each frame even with empty detections.
            NOTE: The number of objects returned may differ from the number of detections provided.
            STRUCTURE of each return track: [xmin, ymin, xmax, ymax, bit_area, cur_time, basket_count, basket_time, local_id]
            bit_area: 1 = ENTER, 2 = EXIT, 3 = A_AREA, 4 = B_AREA, -1 = NOT IN SPECIAL AREA
            cur_time: current timestamp
            basket_count: number frames track has basket
            basket_time: if basket_count > BASKET_FREQ, basket_time is first time track has basket else first time create track
            local_id: ID of track
            frame: current frame
            headpose_Detector: headpose detection model
        """
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self._trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self._trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self._trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self._low_iou_threshold)
        res = np.zeros((len(dets), 6))
        # update matched trackers with assigned detections
        for d, t in matched:
            self._trackers[t].update(dets[d], self._min_hits)
            res[d, 0:4] = dets[d, 0:-1]
            res[d, -1] = self._trackers[t].id
            res[d, -2] = self._timestamp

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i], self._timestamp)
            self._trackers.append(trk)
            res[i, 0:4] = dets[i, 0:-1]
            res[i, -1] = trk.id
            res[i, -2] = self._timestamp

        # Update basket to existing tracks
        tracked_faces = self.associate_faces2trackers(faces)

        # Update accompany people to existing tracks
        self.count_accompany_ppl2trackers(res)
        
        # check the attention 
        self.check_attention(headpose_Detector,tracked_faces,frame)

        # remove dead tracklet
        i = len(self._trackers)
        localIDs_end = []
        for trk in reversed(self._trackers):
            i -= 1
            if (trk.time_since_update > self._max_age):
                ppl_accompany = np.asarray(list(trk.ppl_dist.values()))
                ppl_accompany = ppl_accompany[ppl_accompany > self._min_area_freq]
                # check no attention + calculate the duration
                if trk.cnt_frame_attention > int(os.getenv('THRESHOLD_HEADPOSE')):
                    duration_attention = str(trk.cnt_frame_attention / int(os.getenv('FPS_CAM_SIGNAGE')))
                    duration_group = calculate_duration(trk.basket_time, self._timestamp)

                    # *IMPORTANT NOTE: basket_time: the first time the person appears in the video, just re-use 
                    localIDs_end.append([trk.id, len(ppl_accompany), trk.basket_time, self._timestamp, 'has_attention', trk.start_hp_time, duration_attention, duration_group,trk.end_hp_time])
                else:
                    duration_attention = 'None'
                    duration_group = calculate_duration(trk.basket_time,self._timestamp)
                    localIDs_end.append([trk.id, len(ppl_accompany), trk.basket_time,self._timestamp,'no','None',duration_attention, duration_group,'None'])

                self._trackers.pop(i)

        if (len(res) > 0):
            return res, localIDs_end
        return np.empty((0, 6)), localIDs_end

    def count_accompany_ppl2trackers(self, res):
        for i, trki in enumerate(res):
            for j, trkj in enumerate(res):
                if (j == i) or (trki[-1] < 1) or (trkj[-1] < 1): continue

                if compare_2bboxes_area(trki[:4], trkj[:4], self._min_area_ratio, self._max_area_ratio):
                    for trk in self._trackers:
                        if trk.id == int(trki[-1]):
                            trk.ppl_dist[int(trkj[-1])] = trk.ppl_dist.get(int(trkj[-1]), 0) + 1

    def associate_faces2trackers(self, baskets):
        # get locations from existing trackers.
        trks = np.zeros((len(self._trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self._trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self._trackers.pop(t)

        matched, _, _ = associate_detections_to_trackers(baskets, trks, low_iou_threshold=0.01)
        for t in matched:
            baskets[t[0], -1] = self._trackers[t[1]].id

        return baskets


    def check_attention(self, detector, faces, frame):
        """
            Signage Camera 2: if the face appeared in the frame, immediately consider it as 'has_attention'
            
            Args:
                detector: headpose detector
                faces: bounding boxes of faces [xmin,ymin,xmax,ymax,person_id]
                frame: current frame
            Return:
                None
        """

        # faces : are tracked face, associated with ID already
        for face in faces:
            person_id = int(face[4])
            if person_id < 0: continue
            face_box = face[:4]
            yaw,pitch,roll = detector.getOutput(input_img=frame, box=face_box)

            # draw prediction
            detector.draw_axis(frame, face_box, yaw, pitch, roll,size = 40)

            for index,trk in enumerate(self._trackers):
                if int(trk.id) == person_id:
                    # put all remaining code at here
                    self._trackers[index].look_prediction = 'has_attention'
                    self._trackers[index].head_pose = (yaw, pitch, roll)
                    if self._trackers[index].cnt_frame_attention == 0:
                        self._trackers[index].start_hp_time = self._timestamp
                    self._trackers[index].cnt_frame_attention +=1
                    self._trackers[index].end_hp_time = self._timestamp
