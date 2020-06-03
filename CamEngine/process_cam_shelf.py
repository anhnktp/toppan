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
from helpers.time_utils import convert_to_jp_time
from helpers.shelves_loc_utils import get_shelves_loc, draw_shelves_polygon

from modules.DataTemplate import DataTemplate
from modules.ActionRecognition import ActionRecognition
from modules.PoseExtraction import PoseExtraction
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
        item_boxes.append((shelf, shelf_item_box))
    return item_boxes


def get_start_time(video_fname, cam_type):
    try:
        start_time = get_timestamp_from_filename(video_fname, cam_type)
        print(f"Get start time from {video_fname}")
    except:
        start_time = time.time()
        print("Start time is current time")

    return start_time


def process_cam_shelf(camShelf_queue, cam_type, num_loaded_model, global_tracks):
    engine_logger.critical('------ {} flow process started ------'.format(cam_type))

    video_base_dir = os.getenv('VIDEO_BASE_DIR')
    save_base_dir = os.getenv('SAVE_BASE_DIR')
    exp_name = 'separate_pose'
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
        # videos_input[0]: Take the 01_area1_shelf_right*.mp4
        video_input = videos_input[0] 
    else:
        video_input = videos_input[1] 
    
    vid = cv2.VideoCapture(video_input)
    width, height, fps, num_frames = get_vid_properties(vid)

    basename = video_input.split('/')[-1]
    vw = VideoWriter(width, height, fps, case_save_dir, basename)
    cam_id = int(basename.split('_')[0])

    shelves_item_boxes = get_shelf_item(cam_type)
    item_boxes = [shelves_item_boxes[i][1] for i in range(len(shelves_item_boxes))]
    item_boxes_paint = [np.asanyarray(item_box, np.int32) for item_box in item_boxes]

    # Create pose extraction
    roi_top, roi_bottom = 0, height
    roi_left, roi_right = 0, width
    pose_extraction = PoseExtraction(os.getenv('CAM_SHELF_GPU'), 
                        os.getenv('MODEL_POSE_PATH'), 
                        roi_top, roi_bottom, roi_left, roi_right, 
                        item_boxes)
    
    action = ActionRecognition(cam_type)

    frame_cnt = 0
    old_state = []
    max_age = 5
    
    # Get start timestamp on video
    start_time = get_start_time(os.path.basename(video_input), cam_type)
    
    while vid.isOpened():
        grabbed, img1 = vid.read()
        if not grabbed: break

        cur_time = start_time + vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    
        img_data = dict()
        img_data['frame'] = img1
        img_data['timestamp'] = cur_time
        img_ori = img_data['frame']

        # Get hand keypoints
        hands = pose_extraction.get_hand_coord(img_ori, float(os.getenv('HAND_SCORE')))
        pose_extraction.draw_hand(img_ori, hands)
        
        trackers = []

        # Draw shelf polygons
        for item_box_paint in item_boxes_paint:
            if item_box_paint.shape[0] == 0:
                continue
            cv2.polylines(img_ori, [np.asarray(item_box_paint)], True, (0, 0, 255), thickness=2)

        states = old_state.copy()
        # Touch shelf event
        new_shelves_hand_touched = action.detect_action(states, hands, trackers, shelves_item_boxes)

        if len(old_state) == max_age:
            old_state.pop(0)
        
        old_state.append(hands)

        if len(new_shelves_hand_touched) > 0:
            h_time = convert_timestamp_to_human_time(cur_time)
            print(new_shelves_hand_touched, h_time)
        
        vw.write(img_ori)
    
    vid.release()
    vw.release()


if __name__ == '__main__':
    camShelf_queue = []
    cam_type = 'CAM_SHELF_02'
    num_loaded_model = 1
    global_tracks = []
    
    process_cam_shelf(camShelf_queue, cam_type, num_loaded_model, global_tracks)







 