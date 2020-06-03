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

def get_timestamp_from_filename(filename, cam_type='CAM_360'):
    if cam_type == 'CAM_360':
        YYYY, mm, dd, HH, MM, SS = filename[0:-4].split('_')[3:]
    else:
        YYYY, mm, dd, HH, MM, SS = filename[0:-4].split('_')[4:]
    human_time = '{}:{}:{} {}:{}:{}'.format(YYYY, mm, dd, HH, MM, SS)
    return time.mktime(time.strptime(human_time, '%Y:%m:%d %H:%M:%S'))

def convert_timestamp_to_human_time(unix_timestamp):
    return datetime.datetime.fromtimestamp(unix_timestamp).strftime('%Y:%m:%d %H:%M:%S.%f')[:-2]
