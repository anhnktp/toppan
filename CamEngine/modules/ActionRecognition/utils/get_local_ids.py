from helpers.time_utils import concat_local_id_time

def get_local_ids(cam_type, trackers):
    local_ids = []
    cam_shelf_dict = ['CAM_SHELF_01', 'CAM_SHELF_02']
    for trk in trackers:
        # if local_id = -1 or track in non-shelf area not process
        if (trk[-1] < 0) or (trk[-3] < 0):
            continue
        # if track not in cam_type area not process
        if cam_shelf_dict[int(trk[-3]) - 1] != cam_type:
            continue
        local_id_time = concat_local_id_time(trk[-1], trk[-2])
        local_ids.append(local_id_time)

    local_ids = filter_local_ids(local_ids, cam_type, trackers)
    return local_ids


def filter_local_ids(local_ids, cam_type, trackers):
    # TODO: filter local_ids for each CamID
    return local_ids