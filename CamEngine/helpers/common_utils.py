import base64
import numpy as np
from PIL import Image
import os
import os.path as osp
import cv2
import shutil
import pandas as pd
from shapely.geometry import Point
from scipy.optimize import linear_sum_assignment
from datetime import datetime

try:
    import accimage
except ImportError:
    accimage = None


def cv2_base64encode(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    return base64.b64encode(buffer.tostring())


def cv2_base64decode(base64_string):
    nparr = np.fromstring(base64.b64decode(base64_string), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv2_PIL(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def crop(img, xmin, ymin, xmax, ymax):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((xmin, ymin, xmax, ymax))


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def extract_vid(src_path, out_dir, period=1):
    assert osp.exists(src_path), 'Source file not found !'
    vid = cv2.VideoCapture(src_path)
    if osp.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    cnt = 0
    while vid.isOpened():
        _, img = vid.read()
        if img is None: break
        if cnt % period == 0:
            img_name = '{:05n}.jpg'.format(cnt)
            img_path = osp.join(out_dir, img_name)
            cv2.imwrite(img_path, img)
            cnt += 1
    vid.release()


def get_public_ip():
    from requests import get
    # Get external ip for host machine
    return get('https://api.ipify.org').text


def draw_polygon(img, box, colour=(0, 0, 255), thickness=2):
    '''
        :param img: cv2 image
        :param box: list vertices of box [(x1, y1), (x2, y2, ... , (xn, yn)]
    '''
    cv2.polylines(img, [np.asarray(box, np.int32)], True, colour, thickness=thickness)


class CSV_Writer(object):

    def __init__(self, column_name, csv_path):
        super(CSV_Writer, self).__init__()
        self.csv_data = []
        self.column_name = column_name
        self.csv_path = csv_path

    def write(self, data):
        self.csv_data.append(data)

    def to_csv(self, sep=',', index_label='ID', sort_column=None):
        # Output to csv file
        csv_df = pd.DataFrame(self.csv_data, columns=self.column_name)
        if sort_column: csv_df.sort_values(by=sort_column, inplace=True)
        csv_df.to_csv(self.csv_path, index=True, index_label=index_label, sep=sep)

def filter_by_timestamp(csv_df, query_timestamp, info):
    '''
    :param csv_df: pandas dataframe
           query_timestamp: unix timestamp
    :return: csv_df panda dataframe after filter by csv_df['timestamp] <= query_timestamp and csv_df['timestamp] == info
            and drop duplicate row with same 'shopper ID' and only keep row with nearest unix timestamp with query_timestamp
    '''
    res = csv_df.loc[(csv_df['timestamp'] <= query_timestamp) & (csv_df['info'] == info)]
    return res.drop_duplicates(subset=['shopper ID'], keep='last')

def load_csv(path, col=None):
    return pd.read_csv(path, usecols=col).rename(columns={'timestamp (unix timestamp) ': 'timestamp'})

def load_csv_signage(path):
    csv_signage_df = pd.read_csv(path, parse_dates=[-4], date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    csv_signage_df.rename(columns={'Shopper_ID': 'shopper ID'}, inplace=True)
    csv_signage_df['timestamp'] = csv_signage_df['Timestamp (UTC-JST)'].values.astype(np.int64) / 10 ** 9 - 7*3600 + 1
    return csv_signage_df

def map_id_shelf(trackers, list_shelf_id, shelf_a_area, shelf_ids_xy):
    list_track_info = []
    list_local_id = []
    for trk in trackers:
        if trk[-1] < 0: continue
        center_point = Point((trk[0] + trk[2]) / 2, (trk[1] + trk[3]) / 2)
        if (shelf_a_area.contains(center_point)):
            list_track_info.append([(trk[0] + trk[2]) / 2, (trk[1] + trk[3]) / 2, int(trk[-1])])
            list_local_id.append(int(trk[-1]))
    iou_matrix = np.zeros((len(list_track_info), len(list_shelf_id)), dtype=np.float32)
    for d, trk in enumerate(list_track_info):
        for t, shelf_info in enumerate(list_shelf_id):
            mid_top_point = np.array((trk[0], trk[1]))
            shelf_id_point = np.array(shelf_ids_xy[shelf_info['shelf_id'] - 1])
            iou_matrix[d, t] = np.linalg.norm(mid_top_point - shelf_id_point)
    row_indices, col_indices = linear_sum_assignment(iou_matrix)
    matched_indices = np.column_stack((row_indices, col_indices))
    for d, t in matched_indices:
        list_shelf_id[t]['local_id'] = list_track_info[d][-1]
    for t, shelf_info in enumerate(list_shelf_id):
        if (t not in matched_indices[:, 1]):
            shelf_info['local_id'] = list_local_id

    return list_local_id

def map_id_signage(trackers, sigange_area):
    list_local_id = []
    for trk in trackers:
        if trk[-1] < 0: continue
        center_point = Point((trk[0] + trk[2]) / 2, (trk[1] + trk[3]) / 2)
        if sigange_area.contains(center_point):
            list_local_id.append(int(trk[-1]))
    return list_local_id

def calculate_duration(start,finish):
    '''
        Calculate the duration of looking based on start look timestamp and end look timestamp
        Convert the timestamp to string 
        Convert the string to 
    '''

    # convert the unix - > string
    converted_start = datetime.fromtimestamp(start).strftime('%Y:%m:%d %H:%M:%S.%f')
    converted_finish = datetime.fromtimestamp(finish).strftime('%Y:%m:%d %H:%M:%S.%f')

    # convert once more time 
    converted_start = datetime.strptime(converted_start,'%Y:%m:%d %H:%M:%S.%f')
    converted_finish = datetime.strptime(converted_finish,'%Y:%m:%d %H:%M:%S.%f')

    # print ("Convert finish:", converted_finish)
    # print ("Convert start:", converted_start)

    # caclualte the difference -> convert to seconds
    duration = converted_finish - converted_start
    total_seconds = duration.total_seconds()

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = duration.seconds
    miliseconds = int(duration.microseconds / 1000)

    # human reaable format
    duration = '{}.{}'.format(seconds,miliseconds)

    return duration