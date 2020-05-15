import numpy as np
from helpers.concat_local_id import concat_local_id_time

def check_qr_exit(trackers, QR_EXIT_POS, OFFSET_DISTANCE):
    localIDs_exited = []
    if len(trackers) == 0: return localIDs_exited
    min_distance = 1000000000
    for trk in trackers:
        localID = trk[-1]
        if localID < 0:
            continue

        center = (trk[0]+trk[2])/2, (trk[1]+trk[3])/2
        vector = np.array([center[0] - QR_EXIT_POS[0], center[1] - QR_EXIT_POS[1]])
        distance = np.linalg.norm(vector)
        if min_distance > distance:
            min_distance = distance
            local_id = localID
            timestamp = trk[-2]

    if min_distance < OFFSET_DISTANCE:
        local_id_time = concat_local_id_time(local_id, timestamp)
        localIDs_exited.append(local_id_time)

    return localIDs_exited


