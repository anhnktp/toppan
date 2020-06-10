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
# from modules.PoseExtraction import PoseExtraction
from helpers.time_utils import convert_to_jp_time
from modules.EventManager import EventManager
from modules.VMSManager import VMSManager
from modules.Detection.Detector_hand import HandDetector
from modules.Visualization import HandVisualizer

from modules.Detection.hand import HandCenter, track_hands
from helpers.common_utils import CSV_Writer

def get_start_time(video_fname, cam_type):
    try:
        start_time = get_timestamp_from_filename(video_fname, cam_type)
        print(f"Get start time from {video_fname}")
    except:
        start_time = time.time()
        print("Start time is current time")

    return start_time

def process_cam_shelf(camShelf_queue, cam_type, num_loaded_model, global_tracks, n):
    engine_logger.critical('------ {} flow process started ------'.format(cam_type))
    
    video_base_dir = os.getenv('VIDEO_BASE_DIR')
    save_base_dir = os.getenv('SAVE_BASE_DIR')
    exp_name = 'separate'
    save_base_dir = os.path.join(save_base_dir, exp_name, 'test_cases')
    cases = ['case{}'.format(str(i+1).zfill(2)) for i in range(25)]

    case = cases[n]

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

    vis = HandVisualizer()

    # Hand detector
    detector = HandDetector(os.getenv('HAND_CFG_PATH'),
                            os.getenv('HAND_MODEL_PATH'), 
                            os.getenv('CAM_SHELF_GPU'),
                            os.getenv('HAND_SCORE_THRESHOLD'),
                            os.getenv('HAND_NMS_THRESHOLD'),
                            os.getenv('HAND_BOX_AREA_THRESHOLD'))

    roi_x1y1, roi_x2y2 = (0, 0), (width, height)
    detector.setROI(roi_x1y1, roi_x2y2)

    # Hand action recognition
    action = HandActionRecognition(cam_type)

    previous_hands_center = []
    old_state = []
    max_age = 10

    # Get start timestamp on video
    start_time = get_start_time(os.path.basename(video_input), cam_type)

    handTracker = {}
    setHandId = set()

    new_shelves_hand_touched_list = []

    while vid.isOpened():
        grabbed, img = vid.read()
        if not grabbed: break

        cur_time = start_time + vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Get detection results
        frame = img[roi_x1y1[1]:roi_x2y2[1], roi_x1y1[0]:roi_x2y2[0]]
        detector.setFrame(frame)
        hands = detector.getOutput(cur_time)

        ### Simple hand tracker refering to lightweight openpose
        # current_hand_centers = []
        # for hand in hands:
        #     hand_center = hand[-1]
        #     confidence = hand[-2]
        #     # handcenter object
        #     hc = HandCenter(hand_center, confidence)
        #     current_hand_centers.append(hc)
        #
        # track_hands(previous_hands_center, current_hand_centers)
        # previous_hands_center = current_hand_centers
        # for hc in current_hand_centers:
        #     cv2.putText(img, 'id: {}'.format(hc.id), hc.hand_center,
        #                     cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        ### Simple hand tracker refering t so lightweight openpose

        ### Calculate hand velocity
        current_hand_center = []

        for hand in hands:
            hand_center = hand[-1]
            hand_id = hand[-3]
            hand_time = hand[-2]
            setHandId.add(hand_id)
            for id in setHandId:
                if id not in handTracker.keys():
                    handTracker[id] = []
                if id == hand_id:
                    if len(handTracker[id]) > 1:
                        handTracker[id].pop(0)
                    handTracker[id].append([hand_center, hand_time])

                    if len(handTracker[id]) == 2:
                        try:
                            c1 = handTracker[hand_id][0][0]
                            c2 = handTracker[hand_id][1][0]
                            deltaT = handTracker[hand_id][1][1] - handTracker[hand_id][0][1]
                            velo = (np.linalg.norm(np.array(c1) - np.array(c2)))/deltaT
                            vx = (c2[0] - c1[0])/deltaT
                            vy = (c2[1] - c1[1])/deltaT
                            delta = min(hand[2]-hand[0], hand[3]-hand[1])
                            xc = int(hand[-1][0] + delta*0.3*(vx/abs(vx))*(abs(vx)/velo))
                            yc = int(hand[-1][1] - delta*0.3*(abs(vy)/velo))
                            hand.insert(6, vx)
                            hand.insert(7, vy)
                            hand.insert(8, velo)
                            hand[-1] = (xc, yc)
                            # hand: [xmin, ymin, xmax, ymax, id, time, vx, vy, velo, (xc,yc)]
                            cv2.putText(frame, str(int(velo)) + '_vx'+ str(int(vx)) + '_vy' + str(int(vy)), (c2[0]-50, c2[1]+50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                        except:
                            print('unable to calculate hand velocity')

        trackers = []
        # Shelf touch detection
        states = old_state.copy()
        new_shelves_hand_touched = action.detect_action(states, hands, trackers, item_boxes)
        if len(old_state) == max_age:
            old_state.pop(0)

        old_state.append(hands)

        if len(new_shelves_hand_touched) > 0:
            h_time = convert_timestamp_to_human_time(cur_time)
            #print(new_shelves_hand_touched, h_time)
            shelves = []
            for new_shelves_hand in new_shelves_hand_touched:
                new_shelves_hand = list(new_shelves_hand)
                new_shelves_hand.append(h_time)
                shelves.append(new_shelves_hand)
            new_shelves_hand_touched_list.append(shelves)
        # Visualize handboxes
        vis.draw_boxes(hands, frame)

        # Draw shelves' polygon
        draw_shelves_polygon(frame, shelves_info)

        vw.write(frame)
    print(new_shelves_hand_touched_list)
    return new_shelves_hand_touched_list
    vid.release()
    vw.release()

def merge_csv (rets1, rets2, csv_path):
    rets1_t = []
    for ret in rets1:
        try:
            ret = (1, ret[0][0], ret[0][1], ret[0][2], ret[0][3], ret[0][4])
            rets1_t.append(ret)
        except:
            print('a')
    rets2_t = []
    for ret in rets2:
        try:
            ret = (2, ret[0][0], ret[0][1], ret[0][2], ret[0][3], ret[0][4])
            rets2_t.append(ret)
        except:
            print('b')
    rets = rets1_t
    rets.extend(rets2_t)
    rets.sort(key=lambda tup: tup[4])

    column_name = ['camera ID', 'shelf ID', 'hand_id', 'hand_coords', 'timestamp', 'timestamp(UTC - JST)']
    csv_writer = CSV_Writer(column_name, csv_path)
    for ret in rets:
        csv_writer.write(ret)
    csv_writer.to_csv()


if __name__ == '__main__':
    camShelf_queue = []
    cam_type1 = 'CAM_SHELF_01'
    cam_type2 = 'CAM_SHELF_02'
    num_loaded_model = 1
    global_tracks = []

    for i in range(24,25):
        rets1 = process_cam_shelf(camShelf_queue, cam_type1, num_loaded_model, global_tracks, i)
        rets2 = process_cam_shelf(camShelf_queue, cam_type2, num_loaded_model, global_tracks, i)

        path = 'CSV/' + 'case{}'.format(str(i+1).zfill(2)) + '.csv'
        try:
            merge_csv(rets1, rets2, path)
        except:
            print('unable to merge')













 