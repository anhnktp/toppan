import time
import redis
import ast
import uuid
from collections import deque
import glob

import cv2
import numpy as np
import pandas as pd

from helpers.settings import *
from helpers.file_utils import remove_file
from helpers.video_utils import get_vid_properties, VideoWriter
from helpers.time_utils import *
from helpers.shelves_loc_utils import get_shelves_loc, draw_shelves_polygon

from modules.DataTemplate import DataTemplate
from modules.ActionRecognition import HandActionRecognition
from modules.PoseExtraction import PoseExtraction
from helpers.time_utils import convert_to_jp_time
from modules.EventManager import EventManager
from modules.VMSManager import VMSManager


def get_shelf_item(cam_type):
    if cam_type == 'CAM_SHELF_01':
        shelves = [2, 5]
    else:
        shelves = [1, 3, 4]

    item_boxes = []
    for shelf in shelves:
        cam_shelf_item = 'ITEM_{}_{:02d}'.format(cam_type, shelf)
        shelf_item_box = list(ast.literal_eval(os.getenv(cam_shelf_item)))
        item_boxes.append(shelf_item_box)
    return item_boxes


def get_start_time(video_fname, cam_type):
    try:
        start_time = get_timestamp_from_filename(video_fname, cam_type)
        print(f"Get start time from {video_fname}")
    except:
        start_time = time.time()
        print("Start time is current time")

    return start_time


if __name__ == '__main__':
    cam_type = 'CAM_SHELF_01'
    video_base_dir = os.getenv('VIDEO_BASE_DIR')
    save_base_dir = os.getenv('SAVE_BASE_DIR')
    exp_name = 'separate'
    save_base_dir = os.path.join(save_base_dir, exp_name, 'test_cases')
    cases = ['case{}'.format(str(i+1).zfill(2)) for i in range(25)]

    case = cases[-2]
    case_dir = os.path.join(video_base_dir, case)
    case_save_dir = os.path.join(save_base_dir, case)
    if not os.path.exists(case_save_dir):
        os.makedirs(case_save_dir)

    # Remove a file if it exists.
    # fnames = [os.path.join(case_dir, f'{case}.{ext}') for ext in ['mp4', 'csv']]
    # map(remove_file, fnames)

    videos_input = glob.glob(case_dir+ '/*.mp4')
    videos_input.sort()
    
    if cam_type == 'CAM_SHELF_01':
        shelves_info = get_shelves_loc(os.getenv('CAM_SHELF_01_POLYGONS_ANNO_FILE'))
        # videos_input[0]: Take the 01_area1_shelf_right*.mp4
        video_input = videos_input[0] 
    else:
        shelves_info = get_shelves_loc(os.getenv('CAM_SHELF_02_POLYGONS_ANNO_FILE'))
        video_input = videos_input[1] 
    item_boxes = shelves_info['shelf_dict']
    
    vid = cv2.VideoCapture(video_input)
    width, height, fps, num_frames = get_vid_properties(vid)

    basename = video_input.split('/')[-1]
    vw = VideoWriter(width, height, fps, case_save_dir, basename)
    cam_id = int(basename.split('_')[0])

    item_boxes = get_shelf_item(cam_type)
    # Create pose extraction
    roi_top, roi_bottom = 0, height
    roi_left, roi_right = 0, width
    pose_extraction = PoseExtraction(os.getenv('CAM_SHELF_GPU'), 
                        os.getenv('MODEL_POSE_PATH'), 
                        roi_top, roi_bottom, roi_left, roi_right, 
                        item_boxes)
    vid.release()
    vw.release()








 