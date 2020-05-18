import numpy as np
from shapely.geometry import Point
from helpers.bbox_utils import min_box_iou
from helpers.concat_local_id import concat_local_id_time

def check_overwrap1(prev_trackers, cur_trackers, list_trackers, index_trk_to_del, min_hits, OVERWRAP_IOU_THRESHOLD):
    overwrap_cases = []
    if len(prev_trackers) > 0 and len(cur_trackers) > 0:
        prev_local_id = prev_trackers[:, -1]
        cur_local_id = cur_trackers[:, -1]
    else:
        return overwrap_cases, index_trk_to_del, list_trackers   # equal return []
    if np.in1d(-1, prev_local_id) or np.in1d(-1, cur_local_id):
        is_check = np.empty((0, 2))
        for prev_trk in prev_trackers:
            for cur_trk in cur_trackers:
                # if prev_id = cur_id, continue
                if prev_trk[-4] == cur_trk[-4]: continue
                # if checked_overwrap of (prev_id, cur_id) or (cur_id, prev_id), continue
                check1 = np.where(np.all(is_check == [prev_trk[-4], cur_trk[-4]], axis=1))
                check2 = np.where(np.all(is_check == [cur_trk[-4], prev_trk[-4]], axis=1))
                if (len(check1) > 0) or (len(check2) > 0): continue
                # if iou > threshold
                is_check = np.append(is_check, [[prev_trk[-4], cur_trk[-4]]], axis=0)
                is_check = np.append(is_check, [[cur_trk[-4], prev_trk[-4]]], axis=0)
                if (prev_trk[-1] > 0) and (cur_trk[-1] > 0) and (min_box_iou(prev_trk, cur_trk) > OVERWRAP_IOU_THRESHOLD):
                    overwrap_cases.append([prev_trk[-1], cur_trk[-1]])
                elif (prev_trk[-1] > 0) and (cur_trk[-1] < 0):
                    id_to_del = np.where(index_trk_to_del == prev_trk[-1])[0]
                    if (len(id_to_del) > 0):
                        if min_box_iou(prev_trk, cur_trk) > OVERWRAP_IOU_THRESHOLD:
                            if list_trackers[int(cur_trk[-4])].hit_streak >= min_hits - 1:
                                index_trk_to_del = np.delete(index_trk_to_del, id_to_del, 0)
                                list_trackers.pop(int(cur_trk[-4]))
                        # Consequent
                        else:
                            index_trk_to_del = np.delete(index_trk_to_del, id_to_del, 0)

                    else:
                        index_trk_to_del = np.append(index_trk_to_del, prev_trk[-1])

        return overwrap_cases, index_trk_to_del, list_trackers
    else:
        intersect = np.intersect1d(prev_local_id, cur_local_id)
        union = np.union1d(prev_local_id, cur_local_id)
        if np.array_equal(intersect, union):
            return overwrap_cases, index_trk_to_del, list_trackers  # equal return []
        is_check = np.empty((0, 2))
        for prev_trk in prev_trackers:
            for cur_trk in cur_trackers:
                check1 = np.where(np.all(is_check == [prev_trk[-1], cur_trk[-1]], axis=1))
                check2 = np.where(np.all(is_check == [cur_trk[-1], prev_trk[-1]], axis=1))
                if (len(check1) > 0) or (len(check2) > 0): continue
                is_check = np.append(is_check, [[prev_trk[-1], cur_trk[-1]]], axis=0)
                is_check = np.append(is_check, [[cur_trk[-1], prev_trk[-1]]], axis=0)
                if (prev_trk[-1] != cur_trk[-1]) and (min_box_iou(prev_trk, cur_trk) > OVERWRAP_IOU_THRESHOLD):
                    prev_local_id = concat_local_id_time(prev_trk[-1], prev_trk[-2])
                    local_id = concat_local_id_time(cur_trk[-1], cur_trk[-2])
                    overwrap_cases.append([prev_trk[-1], cur_trk[-1]])
        return overwrap_cases, index_trk_to_del, list_trackers       # equal return []

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