import numpy as np
from helpers.bbox_utils import min_box_iou
from helpers.time_utils import concat_local_id_time

def check_overwrap(prev_trackers, cur_trackers, OVERWRAP_IOU_THRESHOLD):

    overwrap_cases = []
    overwrap_to_draw = []
    if len(prev_trackers) > 0 and len(cur_trackers) > 0:
        prev_trks = prev_trackers[np.where(prev_trackers[:, -1] > 0)[0]]
        cur_trks = cur_trackers[np.where(cur_trackers[:, -1] > 0)[0]]
    else:
        return overwrap_cases, overwrap_to_draw
    if (len(prev_trks) > 0) and (len(cur_trks) > 0):
        prev_trks_local_id = prev_trks[:, -1]
        trks_local_id = cur_trks[:, -1]
        intersect = np.intersect1d(prev_trks_local_id, trks_local_id)
        union = np.union1d(prev_trks_local_id, trks_local_id)
        if np.array_equal(intersect, union):
            return overwrap_cases, overwrap_to_draw           # equal return []
        is_check = np.empty((0, 2))
        for prev_trk in prev_trks:
            for cur_trk in cur_trks:
                check1 = np.where(np.all(is_check == [prev_trk[-1], cur_trk[-1]], axis=1))[0]
                check2 = np.where(np.all(is_check == [cur_trk[-1], prev_trk[-1]], axis=1))[0]
                if (len(check1) > 0) or (len(check2) > 0):
                    continue
                is_check = np.append(is_check, [[prev_trk[-1], cur_trk[-1]]], axis=0)
                is_check = np.append(is_check, [[cur_trk[-1], prev_trk[-1]]], axis=0)
                if (prev_trk[-1] != cur_trk[-1]) and (min_box_iou(prev_trk, cur_trk) > OVERWRAP_IOU_THRESHOLD):
                    prev_local_id = concat_local_id_time(prev_trk[-1], prev_trk[-2])
                    local_id = concat_local_id_time(cur_trk[-1], cur_trk[-2])
                    overwrap_cases.append([prev_local_id, local_id])
                    overwrap_to_draw.append([int(prev_trk[-1]), int(cur_trk[-1])])
    return overwrap_cases, overwrap_to_draw