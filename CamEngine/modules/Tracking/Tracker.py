import numpy as np
from shapely.geometry import Point
from .TrackerBase import TrackerBase
from .utils.KalmanBox import KalmanBoxTracker
from .utils.associate_dets_trks import associate_detections_to_trackers
from .utils.check_accompany import compare_2bboxes_area
from helpers.common_utils import calculate_duration
import os
import ast

def find_area(bbox, in_door_box, out_door_box, a_box, b_box, sig1_box, sig2_box):
    center_point = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    top_left_point = Point(bbox[0], bbox[1])
    bottom_left_point = Point(bbox[0], bbox[3])

    # Check if a track is in the singage FoV
    is_in_sig1_area = sig1_box.contains(center_point)
    is_in_sig2_area = sig2_box.contains(center_point)

    if out_door_box.contains(bottom_left_point) or out_door_box.contains(top_left_point):
    # if out_door_box.contains(bottom_left_point):
        return 'OUT_DOOR_AREA', is_in_sig1_area, is_in_sig2_area
    if in_door_box.contains(bottom_left_point) or in_door_box.contains(top_left_point):
    # if in_door_box.contains(bottom_left_point):
        return 'IN_DOOR_AREA', is_in_sig1_area, is_in_sig2_area
    if a_box.contains(center_point):
        return 'A_AREA', is_in_sig1_area, is_in_sig2_area
    if b_box.contains(center_point):
        return 'B_AREA', is_in_sig1_area, is_in_sig2_area
    return None, is_in_sig1_area, is_in_sig2_area

class Tracker(TrackerBase):
    """
       Using SORT Tracking
    """
    def __init__(self, max_age=20, min_hits=5, low_iou_threshold=0.25):
        """
        Sets key parameters for SORT
        """
        super(Tracker, self).__init__()
        self._max_age = max_age
        self._min_hits = min_hits
        self._low_iou_threshold = low_iou_threshold
        self._trackers = []

    def update(self, dets, *args):

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
        baskets, in_door_box, out_door_box, a_box, b_box, none_box, sig1_box, sig2_box = args
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
        res = np.zeros((len(dets), 13))
        # update matched trackers with assigned detections
        for d, t in matched:
            area, is_in_sig1_area, is_in_sig2_area = find_area(dets[d, 0:-1], in_door_box, out_door_box, a_box, b_box, sig1_box, sig2_box)
            if (area is not None) and (self._trackers[t].area != area):
                if (self._trackers[t].area == 'OUT_DOOR_AREA') and (area == 'IN_DOOR_AREA'): res[d, -5] = 1  # 1 = ENTER
                if (self._trackers[t].area == 'IN_DOOR_AREA') and (area == 'OUT_DOOR_AREA'): res[d, -5] = 2  # 2 = EXIT
                if (area == 'A_AREA'): res[d, -5] = 3  # 3 = A
                if (area == 'B_AREA'): res[d, -5] = 4  # 4 = B
                self._trackers[t].area = area
            else:
                res[d, -3] = -1  # -1 = None move to special area
            self._trackers[t].update(dets[d], self._min_hits)

            if is_in_sig1_area:
                if self._trackers[t].sig1_start_time == None:
                    self._trackers[t].sig1_start_time = self._timestamp
                self._trackers[t].sig1_end_time = self._timestamp

            if is_in_sig2_area:
                if self._trackers[t].sig2_start_time == None:
                    self._trackers[t].sig2_start_time = self._timestamp
                self._trackers[t].sig2_end_time = self._timestamp

            res[d, 0:4] = dets[d, 0:-1]
            res[d, -1] = self._trackers[t].id
            res[d, -2] = self._trackers[t].basket_time
            res[d, -3] = self._trackers[t].basket_count
            res[d, -4] = self._timestamp
            res[d, -6] = self._trackers[t].sig1_start_time
            res[d, -7] = self._trackers[t].sig1_end_time
            res[d, -8] = self._trackers[t].sig2_start_time
            res[d, -9] = self._trackers[t].sig2_end_time

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
            res[i, -6] = trk.sig1_start_time
            res[i, -7] = trk.sig1_end_time
            res[i, -8] = trk.sig2_start_time
            res[i, -9] = trk.sig2_end_time

            bbox = dets[i, 0:-1]
            area, is_in_sig1_area, is_in_sig2_area = find_area(bbox, in_door_box, out_door_box, a_box, b_box, sig1_box,
                                                               sig2_box)

            if is_in_sig1_area:
                if trk.sig1_start_time is None:
                    trk.sig1_start_time = self._timestamp
                trk.sig1_end_time = self._timestamp

            if is_in_sig2_area:
                if trk.sig2_start_time is None:
                    trk.sig2_start_time = self._timestamp
                trk.sig2_end_time = self._timestamp

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

        # remove dead tracklet
        i = len(self._trackers)
        localIDs_end = []
        for trk in reversed(self._trackers):
            i -= 1
            if (trk.time_since_update > self._max_age):
                localIDs_end.append([trk.id, trk.basket_count, trk.basket_time, trk.sig1_start_time, trk.sig1_end_time,
                                     trk.sig2_start_time, trk.sig2_end_time, self._timestamp])
                self._trackers.pop(i)

        if (len(res) > 0):
            return res, localIDs_end
        return np.empty((0, 13)), localIDs_end

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
        for t in unmatched_dets:
            baskets[t, -1] = -1     # basket not assigned has id = -1

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

    def update(self, dets, *args):

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
        faces, headpose_Detector, frame = args
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
        self.check_attention(headpose_Detector, tracked_faces, frame)

        # remove dead tracklet
        i = len(self._trackers)
        localIDs_end = []
        for trk in reversed(self._trackers):
            i -= 1
            if (trk.time_since_update > self._max_age):
                ppl_accompany = np.asarray(list(trk.ppl_dist.values()))
                ppl_accompany = ppl_accompany[ppl_accompany > self._min_area_freq]

                # check no attention + calculate the duration
                if len(trk.duration_hp_list) != 0:
                    duration_group = calculate_duration(trk.basket_time, self._timestamp)
                    # *IMPORTANT NOTE: basket_time: the first time the person appears in the video, just re-use
                    localIDs_end.append([trk.id, len(ppl_accompany), trk.basket_time, self._timestamp, 'has_attention',
                                         trk.start_hp_list, trk.duration_hp_list, duration_group, trk.end_hp_list,
                                         trk.sig_start_bbox[0], trk.sig_start_bbox[2],
                                         int(min(max(trk.get_last_state(-1)[0][0], 0),
                                                 ast.literal_eval(os.getenv('IMG_SIZE_CAM_SIGNAGE'))[0])),
                                         int(min(max(trk.get_last_state(-1)[0][2], 0),
                                                 ast.literal_eval(os.getenv('IMG_SIZE_CAM_SIGNAGE'))[0]))])
                else:
                    duration_attention = 'None'
                    duration_group = calculate_duration(trk.basket_time, self._timestamp)
                    localIDs_end.append([trk.id, len(ppl_accompany), trk.basket_time, self._timestamp, 'no', 'None',
                                         duration_attention, duration_group, 'None', trk.sig_start_bbox[0],
                                         trk.sig_start_bbox[2],
                                         int(min(max(trk.get_last_state(-1)[0][0], 0),
                                                 ast.literal_eval(os.getenv('IMG_SIZE_CAM_SIGNAGE'))[0])),
                                         int(min(max(trk.get_last_state(-1)[0][2], 0),
                                                 ast.literal_eval(os.getenv('IMG_SIZE_CAM_SIGNAGE'))[0]))])
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

    def associate_faces2trackers(self, faces):
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

        matched, _, _ = associate_detections_to_trackers(faces, trks, low_iou_threshold=0.01)
        for t in matched:
            faces[t[0], -1] = self._trackers[t[1]].id

        return faces

    def is_valid_headpose(self,yaw,pitch,roll):
        """
            Check the head pose condition based on 3 angles
        """
        if ((yaw > -20.5) & (yaw <20.5) & (roll > -20.5) & (roll < 20.5)):
            return True
        return False

    def check_attention(self, detector, faces, frame):
        """
            Signage Camera 1: Check the headpose angles 
            Signage Camera 2: if the face appeared in the frame, immediately consider it as 'has_attention'. No check headpose angles

            Args:
                detector: headpose detector
                faces: bounding boxes of faces [xmin,ymin,xmax,ymax,person_id]
                frame: current frame
            Return:
                None
        """

        for index, trk in enumerate(self._trackers):
            if trk.id < 0: continue

            face_box = [face[:4] for face in faces if int(face[4]) == int(trk.id)]

            if len(face_box) != 0:
                # calculate yaw,pitch, roll
                yaw, pitch, roll = detector.getOutput(input_img=frame, box=face_box[0])
                # draw the prediction
                detector.draw_axis(frame, face_box[0], yaw, pitch, roll, size=40)

                if (os.getenv('SIGNAGE_ID') == '1' and self.is_valid_headpose(yaw, pitch, roll)) or (
                        os.getenv('SIGNAGE_ID') == '2'):
                    if self._trackers[index].hp_max_age > 0 and self._trackers[index].hp_max_age < int(os.getenv('MAX_AGE_HP')):
                        self._trackers[index].hp_max_age = 0
                    if self._trackers[index].cnt_frame_attention == 0:
                        self._trackers[index].attention = True
                        self._trackers[index].start_hp_time = self._timestamp
                        self._trackers[index].start_hp_list.append(self._timestamp)
                    self._trackers[index].cnt_frame_attention += 1
                    self._trackers[index].end_hp_time = self._timestamp

            else:
                if self._trackers[index].attention:
                    self._trackers[index].hp_max_age += 1
                    if self._trackers[index].hp_max_age > int(os.getenv('MAX_AGE_HP')):
                        # update the duration + reset the state
                        self._trackers[index].end_hp_time = self._timestamp
                        self._trackers[index].end_hp_list.append(self._timestamp)
                        self._trackers[index].duration_hp_list.append(
                            str(trk.cnt_frame_attention / int(os.getenv('FPS_CAM_SIGNAGE'))))
                        self._trackers[index].attention = False
                        self._trackers[index].hp_max_age = 0
                        self._trackers[index].cnt_frame_attention = 0