import numpy as np
from shapely.geometry import Point
from .TrackerBase import TrackerBase
from helpers.concat_local_id import concat_local_id_time
from .utils.KalmanBox import KalmanBoxTracker
from .utils.associate_dets_trks import associate_detections_to_trackers

def find_area(bbox, in_door_box, out_door_box, a_box, b_box):
    center_point = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    top_left_point = Point(bbox[0], bbox[1])
    bottom_left_point = Point(bbox[0], bbox[3])
    if in_door_box.contains(bottom_left_point) or in_door_box.contains(top_left_point):
    # if in_door_box.contains(bottom_left_point):
        return 'IN_DOOR_AREA'
    if out_door_box.contains(bottom_left_point) or out_door_box.contains(top_left_point):
    # if out_door_box.contains(bottom_left_point):
        return 'OUT_DOOR_AREA'
    if a_box.contains(center_point):
        return 'A_AREA'
    if b_box.contains(center_point):
        return 'B_AREA'
    return None

class Tracker(TrackerBase):
    """
       Using SORT Tracking
    """
    def __init__(self, max_age, min_hits, low_iou_threshold=0.25):
        """
        Sets key parameters for SORT
        """
        super(Tracker, self).__init__()
        self._max_age = max_age
        self._min_hits = min_hits
        self._low_iou_threshold = low_iou_threshold
        self._trackers = []

    def update(self, dets, in_door_box, out_door_box, a_box, b_box, none_box):

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
                    if (trk.area == 'IN_DOOR_AREA') and (area == 'OUT_DOOR_AREA'): res[d[0], -3] = 2    # 2 = EXIT
                    if (area == 'A_AREA'): res[d[0], -3] = 3   # 3 = A
                    if (area == 'B_AREA'): res[d[0], -3] = 4   # 4 = B
                    trk.area = area
                else: res[d[0], -3] = -1    # -1 = None move to special area
                if is_new_local_id:
                    trks_start.append(res[d[0]])

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
                res[i, -3] = -1  # -1 = None move to special area

        i = len(self._trackers)
        # remove dead tracklet
        localIDs_end = []
        for trk in reversed(self._trackers):
            i -= 1
            if (trk.time_since_update > self._max_age):
                # last_state = trk.get_last_state(-self._max_age)
                local_id_time = concat_local_id_time(trk.id, trk.timestamp)
                localIDs_end.append(local_id_time)
                self._trackers.pop(i)
        if (len(res) > 0):
            return res, np.asarray(trks_start), localIDs_end
        return np.empty((0, 8)), np.asarray(trks_start), localIDs_end

class Tracker1(TrackerBase):
    """
       Using SORT Tracking
    """
    def __init__(self, max_age, min_hits, low_iou_threshold=0.25):
        """
        Sets key parameters for SORT
        """
        super(Tracker, self).__init__()
        self._max_age = max_age
        self._min_hits = min_hits
        self._low_iou_threshold = low_iou_threshold
        self._trackers = []

    def update(self, dets, in_door_box, out_door_box, a_box, b_box, none_box):

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