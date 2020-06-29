import time
import datetime


def concat_local_id_date(local_id, timestamp):
    date_time = time.strftime('%Y%m%d', time.gmtime(timestamp))
    local_id_date = '_'.join([str(int(local_id)), date_time])
    return local_id_date

def concat_local_id_time(local_id, timestamp):
    local_id_time = '_'.join([str(int(local_id)), str(int(timestamp))])
    return local_id_time

def convert_to_jp_time(utc_timestamp):
    return datetime.datetime.fromtimestamp(int(utc_timestamp), datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y/%m/%d %H:%M:%S")

def get_timestamp_from_filename(filename):
    YYYY, mm, dd, HH, MM, SS = filename[0:-4].split('_')[3:]
    human_time = '{}:{}:{} {}:{}:{}'.format(YYYY, mm, dd, HH, MM, SS)
    return time.mktime(time.strptime(human_time, '%Y:%m:%d %H:%M:%S'))

def convert_timestamp_to_human_time(unix_timestamp):
    return datetime.datetime.fromtimestamp(unix_timestamp).strftime('%Y:%m:%d %H:%M:%S.%f')[:-4]

def compute_time_iou(cam_360_start_time, cam_360_end_time, cam_signage_start_time, cam_signage_end_time):
    """ Compute time IoU between cam_360 and cam_signage
    Args:
        - cam_360_start_time (datetime): start time of a tracker in cam 360
        - cam_360_end_time (datetime): end time of a tracker in cam 360
        - cam_signage_start_time (datetime): start time of a tracker in cam signage
        - cam_signage_end_time (datetime): end time of a tracker in cam signage
    Returns:
        - Time IoU of a tracker between cam 360 and cam signage (0<= IoU <=1)
    """
    if cam_360_end_time <= cam_signage_start_time or cam_360_start_time >= cam_signage_end_time:
        return 0, 0
    else:
        intersection = (min(cam_360_end_time, cam_signage_end_time) - max(cam_signage_start_time, cam_360_start_time)).total_seconds()
        union = (max(cam_signage_end_time, cam_360_end_time) - min(cam_360_start_time, cam_signage_start_time)).total_seconds()
        return intersection / union, intersection