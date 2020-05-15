import time
import datetime


def concat_local_id_date(local_id, timestamp):
    date_time = time.strftime('%Y%m%d', time.gmtime(timestamp))
    local_id_date = '_'.join([str(int(local_id)), date_time])
    return local_id_date

def concat_local_id_time(local_id, timestamp):
    local_id_time = '_'.join([str(int(local_id)), str(int(timestamp))])
    return local_id_time

def convert_to_jp_time(unix_timestamp):
    return datetime.datetime.fromtimestamp(int(unix_timestamp), datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y/%m/%d %H:%M:%S")
