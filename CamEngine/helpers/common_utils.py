import base64
import numpy as np
import os
import os.path as osp
import time
import cv2
import shutil
import pandas as pd
from PIL import Image
from shapely.geometry import Point
from scipy.optimize import linear_sum_assignment
from datetime import datetime
from .time_utils import compute_time_iou

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

def to_csv(csv_path, sep=',', index_label='ID', sort_column=None, csv_df=None):
    # Output to csv file
    if sort_column: csv_df.sort_values(by=sort_column, inplace=True)
    csv_df.to_csv(csv_path, index=True, index_label=index_label, sep=sep)


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
    return pd.read_csv(path, usecols=col)

def load_csv_signage(path, post_processed=False):
    if post_processed:
        return pd.read_csv(path, parse_dates=[-3,-2], date_parser=lambda x: pd.datetime.strptime(x, '%Y:%m:%d %H:%M:%S.%f'))
    else:
        return pd.read_csv(path, parse_dates=[-7,-6], date_parser=lambda x: pd.datetime.strptime(x, '%Y:%m:%d %H:%M:%S.%f'))

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
    duration = finish - start 
    return "%.2f" % duration

def map_local_id(list_local_id, matched_tracks, garbage_tracks):
    list_local_id = list(set(list_local_id) - set(garbage_tracks))
    for i in range(0, len(list_local_id)):
        for track_id, concated_tracks in matched_tracks.items():
            if list_local_id[i] in concated_tracks:
                list_local_id[i] = track_id
                break

    return list(set(list_local_id))

def update_camera_id(filename):
    """ Update the camera id based on the filename """
    if 'signage2' in os.path.basename(filename).split('_'):
        os.environ['SIGNAGE_ID']='2'
    else:
        os.environ['SIGNAGE_ID']='1'

    return int(os.environ['SIGNAGE_ID'])


def post_processing_signage_csv(input_csv, output_csv, time_diff_threshold=1.0, x_overlap_threshold=0.2,
                                accompany_iou_threshold=0.2):
    """ Do post-processing on signage dataframe
    Args:
        - input_csv: input path of the csv file
        - output_csv: path to save the post-processed csv
        - time_diff_threshold: threshold to determine if shoppers are of the same ID
        - x_overlap_threshold: threshold to determine if shoppers are of the same ID
    Returns:
        - post-processed signage dataframe
    """
    # Get signage dataframe from csv
    signage_df = load_csv_signage(input_csv, post_processed=False)
    group_df = signage_df.loc[(signage_df['info'].str.contains('GROUP'))]

    # Find and assign rows of the same shopper
    for index1, row1 in group_df.iterrows():
        for index2, row2 in group_df.iterrows():
            if index1 == index2: continue

            time_diff = (row2['Start_time'] - row1['End_time']).total_seconds()
            x_pos_overlap = compute_x_iou((float(row1['End_bbox_xmin']), float(row1['End_bbox_xmax'])),
                                          (float(row2['Start_bbox_xmin']), float(row2['Start_bbox_xmax'])))

            if time_diff <= time_diff_threshold and time_diff >= 0 and x_pos_overlap >= x_overlap_threshold:
                mask = signage_df['shopper ID'] == signage_df.at[signage_df.index.values[index2], 'shopper ID']
                signage_df['shopper ID'][mask] = signage_df.at[signage_df.index.values[index1], 'shopper ID']

    # Concatenate rows of same shopper ID
    unique_ids = signage_df['shopper ID'].unique()
    concat_df = pd.DataFrame(
        columns=['camera ID', 'shopper ID', 'process ID', 'info', 'Start_time', 'End_time', 'Duration'])

    for identity in unique_ids:
        # Get accompany number
        accompany_df = signage_df.loc[signage_df['info'].str.contains('GROUP') & (signage_df['shopper ID'] == identity)]
        id_start_time = min(accompany_df['Start_time'])
        id_end_time = max(accompany_df['End_time'])
        num_accompany = 1

        for identity2 in unique_ids:
            if identity == identity2: continue
            accompany2_df = signage_df.loc[
                signage_df['info'].str.contains('GROUP') & (signage_df['shopper ID'] == identity2)]
            id2_start_time = min(accompany2_df['Start_time'])
            id2_end_time = max(accompany2_df['End_time'])
            time_iou = compute_time_iou(id_start_time, id_end_time, id2_start_time, id2_end_time)
            if time_iou >= accompany_iou_threshold:
                num_accompany += 1

        duration = (max(accompany_df['End_time']) - min(accompany_df['Start_time'])).total_seconds()
        concat_df = concat_df.append({'camera ID': 1, 'shopper ID': identity, 'process ID': 1203,
                                      'info': 'GROUP {} PEOPLE'.format(num_accompany),
                                      'Start_time': min(accompany_df['Start_time']),
                                      'End_time': max(accompany_df['End_time']), 'Duration': duration},
                                     ignore_index=True)

        # Get attention
        attention_df = signage_df.loc[
            signage_df['info'].str.contains('has_attention') & (signage_df['shopper ID'] == identity)]
        for _, row in attention_df.iterrows():
            concat_df = concat_df.append(
                {'camera ID': 1, 'shopper ID': identity, 'process ID': 1557, 'info': 'has_attention',
                 'Start_time': row['Start_time'], 'End_time': row['End_time'], 'Duration': row['Duration(s)']},
                ignore_index=True)

    concat_df['Start_time'] = concat_df['Start_time'].dt.strftime('%Y:%m:%d %H:%M:%S.%f')
    concat_df['End_time'] = concat_df['End_time'].dt.strftime('%Y:%m:%d %H:%M:%S.%f')

    concat_df.to_csv(output_csv, sep=',', index_label='ID', index=True)


def compute_x_iou(point1_end, point2_start):
    """ Compute overlap in x coordinates of point1 and point2
    Args:
        - point1_end: ending bbox of track 1 (p1_xmin, p1_xmax)
        - point2_start: starting bbox of track 2 (p2_xmin, p2_xmax)
    Returns:
        - IoU in range [0,1]
    """
    if point1_end[1] <= point2_start[0] or point1_end[0] >= point2_start[1]:
        return 0
    else:
        intersection = min(point1_end[1], point2_start[1]) - max(point1_end[0], point2_start[0])
        union = max(point1_end[1], point2_start[1]) - min(point1_end[0], point2_start[0])
        return intersection / union


def combine_signages_to_fisheye(fisheye_df, signage1_df, signage2_df, iou_threshold=0.1):
    """ Combine signage csv to fisheye csv
    Args:
        - fisheye_df (DataFrame): post-processed fisheye csv
        - signage1_df (DataFrame): post-processed signage1 csv
        - signage2_df (DataFrame)L post-processed signage2 csv
    Returns:
        - fisheye_df (DataFrame):
    """

    # Pre-process fisheye camera
    unique_fisheye_ids = fisheye_df['shopper ID'].unique()

    fisheye_sig1_tracks = []
    fisheye_sig2_tracks = []

    for identity in unique_fisheye_ids:
        enter_sig1_df = fisheye_df.loc[
            fisheye_df['info'].str.contains('ENTER SIGNAGE 1') & (fisheye_df['shopper ID'] == identity)]
        leave_sig1_df = fisheye_df.loc[
            fisheye_df['info'].str.contains('LEAVE SIGNAGE 1') & (fisheye_df['shopper ID'] == identity)]

        enter_sig2_df = fisheye_df.loc[
            fisheye_df['info'].str.contains('ENTER SIGNAGE 2') & (fisheye_df['shopper ID'] == identity)]
        leave_sig2_df = fisheye_df.loc[
            fisheye_df['info'].str.contains('LEAVE SIGNAGE 2') & (fisheye_df['shopper ID'] == identity)]

        if len(enter_sig1_df) > 0 and len(leave_sig1_df) > 0:
            fisheye_sig1_tracks.append((identity, datetime.strptime(
                fisheye_df.at[enter_sig1_df.index.values[0], 'timestamp (UTC - JST)'], "%Y:%m:%d %H:%M:%S.%f"),
                                        datetime.strptime(
                                            fisheye_df.at[leave_sig1_df.index.values[0], 'timestamp (UTC - JST)'],
                                            "%Y:%m:%d %H:%M:%S.%f")))
        if len(enter_sig2_df) > 0 and len(leave_sig2_df) > 0:
            fisheye_sig2_tracks.append((identity, datetime.strptime(
                fisheye_df.at[enter_sig2_df.index.values[0], 'timestamp (UTC - JST)'], "%Y:%m:%d %H:%M:%S.%f"),
                                        datetime.strptime(
                                            fisheye_df.at[leave_sig2_df.index.values[0], 'timestamp (UTC - JST)'],
                                            "%Y:%m:%d %H:%M:%S.%f")))

    # Pre-process signage cameras
    signage1_tracks = []
    signage2_tracks = []
    unique_signage1_ids = signage1_df['shopper ID'].unique()
    unique_signage2_ids = signage2_df['shopper ID'].unique()

    for identity in unique_signage1_ids:

        sig1_df = signage1_df.loc[signage1_df['info'].str.contains('GROUP') & (signage1_df['shopper ID'] == identity)]
        if len(sig1_df) > 0:
            signage1_tracks.append((identity, sig1_df.at[sig1_df.index.values[0], 'Start_time'],
                                    sig1_df.at[sig1_df.index.values[0], 'End_time']))

    for identity in unique_signage2_ids:
        sig2_df = signage2_df.loc[signage2_df['info'].str.contains('GROUP') & (signage2_df['shopper ID'] == identity)]
        if len(sig2_df) > 0:
            signage2_tracks.append((identity, sig2_df.at[sig2_df.index.values[0], 'Start_time'],
                                    sig2_df.at[sig2_df.index.values[0], 'End_time']))

    # Compute ious and do id assignment between cam 360 tracks and cam signage 1 tracks
    if len(fisheye_sig1_tracks) > 0 and len(signage1_tracks) > 0:
        IoU_mat = np.zeros((len(fisheye_sig1_tracks), len(signage1_tracks)), dtype=np.float32)
        for r, trk1 in enumerate(fisheye_sig1_tracks):
            for c, trk2 in enumerate(signage1_tracks):
                IoU_mat[r, c] = compute_time_iou(trk1[1], trk1[2], trk2[1], trk2[2])

        matched_row, matched_col = linear_sum_assignment(-IoU_mat)
        for r, c in zip(matched_row, matched_col):
            if IoU_mat[r, c] >= iou_threshold:
                cur_sig_df = signage1_df.loc[(signage1_df['shopper ID'] == signage1_tracks[c][0])]
                for _, row in cur_sig_df.iterrows():
                    process_ID = None
                    info = None
                    if 'GROUP' in row['info']:
                        process_ID = 1203
                        info = row['info'] + ' IN SIGNAGE_1'
                    elif 'attention' in row['info']:
                        process_ID = 1557
                        info = 'HAS ATTENTION TO SIGNAGE_1 IN {}s'.format(row['Duration'])
                    fisheye_df = fisheye_df.append(
                        {'camera ID': 1, 'shopper ID': fisheye_sig1_tracks[r][0], 'process ID': process_ID,
                         'info': info,
                         'timestamp (unix timestamp)': time.mktime(row['Start_time'].timetuple()),
                         'timestamp (UTC - JST)': row['Start_time']}, ignore_index=True)

    # Compute ious and do id assignment between cam 360 tracks and cam signage 1 tracks
    if len(fisheye_sig2_tracks) > 0 and len(signage2_tracks) > 0:
        IoU_mat = np.zeros((len(fisheye_sig2_tracks), len(signage2_tracks)), dtype=np.float32)
        for r, trk1 in enumerate(fisheye_sig2_tracks):
            for c, trk2 in enumerate(signage2_tracks):
                IoU_mat[r, c] = compute_time_iou(trk1[1], trk1[2], trk2[1], trk2[2])

        matched_row, matched_col = linear_sum_assignment(-IoU_mat)
        for r, c in zip(matched_row, matched_col):
            if IoU_mat[r, c] >= iou_threshold:
                cur_sig_df = signage2_df.loc[(signage2_df['shopper ID'] == signage2_tracks[c][0])]
                for _, row in cur_sig_df.iterrows():
                    process_ID = None
                    info = None
                    if 'GROUP' in row['info']:
                        process_ID = 1203
                        info = row['info'] + ' IN SIGNAGE_2'
                    elif 'attention' in row['info']:
                        process_ID = 1557
                        info = 'HAS ATTENTION TO SIGNAGE_2 IN {}s'.format(row['Duration'])
                    fisheye_df = fisheye_df.append(
                        {'camera ID': 1, 'shopper ID': fisheye_sig2_tracks[r][0], 'process ID': process_ID,
                         'info': info,
                         'timestamp (unix timestamp)': time.mktime(row['Start_time'].timetuple()),
                         'timestamp (UTC - JST)': row['Start_time']}, ignore_index=True)
    return fisheye_df.loc[~fisheye_df['info'].str.contains('SIGNAGE ')]