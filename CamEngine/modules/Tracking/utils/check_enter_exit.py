import numpy as np
from shapely.geometry import Point
from helpers.concat_local_id import concat_local_id_time


def check_enter_exit(trackers, queue_enter, entrance_box, camera, trks_start):
    def check_direction(first_bb, last_bb, camera):
        camera_x, camera_y = camera
        vecto_first = np.array([(first_bb[0] + first_bb[2]) / 2 - camera_x,
                                (first_bb[1] + first_bb[3]) / 2 - camera_y])
        vecto_last = np.array([(last_bb[0] + last_bb[2]) / 2 - camera_x,
                               (last_bb[1] + last_bb[3]) / 2 - camera_y])
        if (np.linalg.norm(vecto_first) > np.linalg.norm(vecto_last)):
            return 1        # Enter
        else:
            return 0        # Exit

    def check_in_area(area, trk):
        foot_left = Point(trk[0] + (trk[2] - trk[0])/8, trk[3])
        foot_right = Point(trk[0] + (trk[2] - trk[0])/6, trk[3])
        if area.contains(foot_left) or area.contains(foot_right):
            return 1  # In polygon
        else:
            return 0  # Not in polygon

    K = 5  # check Detection in K frames
    localIDs_entered = []
    if len(trks_start) > 0:
        ids_start = trks_start[:, -1]
    else:
        ids_start = []
    if len(queue_enter) > 0:
        ids_enter = queue_enter[:, -1]  # id of bboxes in queue_enter
    else:
        ids_enter = []
    for trk in trackers:
        localID = trk[-1]
        if localID < 0:
            continue
        if check_in_area(entrance_box, trk):
            if localID in ids_start:
                queue_enter = np.append(queue_enter, [trk], axis=0)
                ids_enter = queue_enter[:, -1]
            elif localID in ids_enter:
                index_track_in_queue_enter = np.where(queue_enter[:, -1] == localID)[0]
                if len(index_track_in_queue_enter) >= K:
                    queue_enter = np.delete(queue_enter, index_track_in_queue_enter[-1], axis=0)
                queue_enter = np.append(queue_enter, [trk], axis=0)

        elif localID in ids_enter:
            trks_localID_in_queue_enter = queue_enter[queue_enter[:, -1] == localID]
            if (len(trks_localID_in_queue_enter) >= K) and check_direction(trks_localID_in_queue_enter[0],
                                                                           trks_localID_in_queue_enter[-1], camera):
                local_id_date = concat_local_id_time(localID, trk[-2])
                queue_enter = np.delete(queue_enter, np.where(queue_enter[:, -1] == localID), 0)
                localIDs_entered.append(local_id_date)
                print('A person entered !')

    return localIDs_entered, queue_enter
