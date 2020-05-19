import time
import redis
import cv2
import ast
import uuid
import numpy as np
from collections import deque
from helpers.settings import *
from modules.DataTemplate import DataTemplate
from modules.ActionRecognition import ActionRecognition
from modules.PoseExtraction import PoseExtraction
from helpers.time_utils import convert_to_jp_time
from modules.EventManager import EventManager
from modules.VMSManager import VMSManager
import pandas as pd


def get_shelf_item(cam_type):
    item_boxes = []
    for shelf in range(5):
        cam_shelf_item = 'ITEM_{}_{:02d}'.format(cam_type, shelf+1)
        shelf_item_box = list(ast.literal_eval(os.getenv(cam_shelf_item)))
        item_boxes.append(shelf_item_box)
    return item_boxes

def process_cam_shelf(camShelf_queue, cam_type, num_loaded_model, global_tracks):

    engine_logger.critical('------ {} flow process started ------'.format(cam_type))

    # Write video
    if os.getenv('SAVE_VID') == 'TRUE':
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        vid_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), '{}.mp4'.format(cam_type))
        videoWriter = cv2.VideoWriter(vid_path, fourcc, int(os.getenv('FPS_CAM_IP')), (1280, 720))

    # Create list to store data in csv
    csv_data = []
    PROCESS_ID = int(os.getenv('PROCESS_ID_SHELF'))
    CAMERA_ID = int(os.getenv('CAMERA_ID'))

    # Create instance of VMS Manager
    # vms = VMSManager()

    # Create instance of Event Manager
    # event = EventManager()

    # Create instance of Redis
    r = redis.Redis(os.getenv("REDIS_HOST"), int(os.getenv("REDIS_PORT")))

    # Create pose extraction
    roi_top = float(os.getenv("{}_ROI_TOP".format(cam_type)))
    roi_bottom = float(os.getenv("{}_ROI_BOTTOM".format(cam_type)))

    roi_left = float(os.getenv("{}_ROI_LEFT".format(cam_type)))
    roi_right = float(os.getenv("{}_ROI_RIGHT".format(cam_type)))

    item_boxes = get_shelf_item(cam_type)
    pose_extraction = PoseExtraction(os.getenv('CAM_SHELF_GPU'), os.getenv('MODEL_POSE_PATH'), roi_top,
                                     roi_bottom, roi_left, roi_right, item_boxes)

    action = ActionRecognition(cam_type)
    frame_cnt = 0
    num_model = len(os.getenv('LIST_CAM_ENGINE').split(','))
    num_loaded_model.value += 1
    old_state = []
    max_age = int(os.getenv('MAX_AGE_ITEM_EVENT'))
    engine_frames = deque(maxlen=20)

    item_boxes_paint = [np.asanyarray(item_box, np.int32) for item_box in item_boxes]
    vid = cv2.VideoCapture(os.getenv('RTSP_AREA1_SHELF_RIGHT'))

    while vid.isOpened():
        # if (num_loaded_model.value < num_model) or (camShelf_queue.qsize() <= 0):
        #     continue
        _, img1 = vid.read()
        if not _: break
        img_data = dict()
        img_data['frame'] = img1
        img_data['timestamp'] = time.time()
        start_time = time.time()
        # img_data = camShelf_queue.get()
        img_ori = img_data['frame']
        cur_time = int(img_data['timestamp'])

        # Get keypoints
        hands = pose_extraction.get_hand_coord(img_ori, float(os.getenv('HAND_SCORE')))

        # Retrieve trackers from fish-eye process
        # trackers = global_tracks.from_redis(int(img_data['timestamp'] * 1000) + 500)
        # if trackers is None: trackers = []
        trackers = []

        for item_box_paint in item_boxes_paint:
            if item_box_paint.shape[0] == 0:
                continue
            cv2.polylines(img_ori, [np.asarray(item_box_paint)], True, (0, 0, 255), thickness=2)

        states = old_state.copy()

        new_shelves_hand_touched = action.detect_action(states, hands, trackers, item_boxes)

        if len(old_state) == max_age:
            old_state.pop(0)

        old_state.append(hands)

        # Display the resulting frame
        cv2.putText(img_ori, 'Frame #{:d} ({:.2f}ms)'.format(frame_cnt, (time.time() - start_time) * 1000), (2, 35),
                    0, fontScale=0.6, color=(0, 255, 0), thickness=2)

        # is_draw = [True] * len(localIDs_item)
        # if len(localIDs_item) > 0:
        #     for i, (_, bbox) in enumerate(localIDs_item):
        #         if not is_draw[i]:
        #             continue
        #         list_local_id = []
        #         for _i, (_local_id_time, _bbox) in enumerate(localIDs_item):
        #             if (bbox[0] == _bbox[0]) and \
        #                     (bbox[1] == _bbox[1]) and \
        #                     (bbox[2] == _bbox[2]) and \
        #                     (bbox[3] == _bbox[3]):
        #                 list_local_id.append(_local_id_time)
        #                 is_draw[_i] = False
        #
        #         x1, y1, x2, y2 = bbox
        #         if (x1 == -1) or (y1 == -1):
        #             x1 = 0
        #             y1 = 50
        #         cv2.putText(img_ori, 'ITEM: {}'.format(", ".join(list_local_id)), (x1, y1), 0,
        #                     fontScale=0.9,
        #                     color=(0, 255, 0), thickness=2)

        # Save events to events collection in db
        # save_events_toDB(vms, event, localIDs_item, cur_time, engine_frames, in_shelf)

        engine_logger.info(
            "Frame Count: {} - {} Flow with FPS: {}".format(frame_cnt, cam_type, 1. / (time.time() - start_time)))
        if len(new_shelves_hand_touched) > 0:
            for i in range(len(new_shelves_hand_touched)):
                new_shelves_hand_touched[i] += 1
                jp_time = convert_to_jp_time(cur_time)
                csv_data.append((CAMERA_ID, -1, PROCESS_ID, new_shelves_hand_touched[-1], cur_time, jp_time))

            # new_shelves_hand_touched = [x + 1 for x in new_shelves_hand_touched]
            cv2.putText(img_ori, 'SHELF_ID:{}'.format(", ".join(map(str, new_shelves_hand_touched))), (2, 55), 0, fontScale=0.6,
                        color=(0, 0, 255), thickness=2)
        if os.getenv('SHOW_GUI_SHELF') == 'TRUE':
            cv2.imshow('{} Human Pose Estimation Python Demo'.format(cam_type), img_ori)
            cv2.waitKey(1)

        if os.getenv('SAVE_VID') == 'TRUE':
            videoWriter.write(img_ori)

        frame_cnt += 1

    # Create result csv file
    column_name = ['camera ID', 'shopper ID', 'process ID', 'shelf ID', 'timestamp (unix timestamp)', 'timestamp (UTC - JST)']
    csv_df = pd.DataFrame(csv_data, columns=column_name)
    csv_df.to_csv(os.getenv('CSV_CAM_SHELF'), index=True, index_label='ID')
    engine_logger.info('Created successfully CSV file of CAM_AREA1!')

    cv2.destroyAllWindows()
    if os.getenv('SAVE_VID') == 'TRUE':
        videoWriter.release()

    engine_logger.critical('------ {} flow process stopped ------'.format(cam_type))

def save_events_toDB(vms, event, localIDs_item, cur_time, engine_frames, in_shelf):

    cur_time_in_ms = cur_time * 1000
    # Save ITEM event to database
    if len(localIDs_item) > 0:
        localIDs_item = [local_id for local_id, bbox in localIDs_item]
        query = {'event_type': 'ITEM', 'local_id': localIDs_item, 'timestamp': {'$gt': cur_time - int(os.getenv('ITEM_OFFSET_TIME'))}}
        same_event = event.find_event(query)
        if len(same_event) > 0: return
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_name = '{}.mp4'.format(str(uuid.uuid1()))
        video_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), video_name)
        writer_event = cv2.VideoWriter(video_path, fourcc, 5, (1280, 720))
        for i in range(0, len(engine_frames)):
            writer_event.write(engine_frames[i])
        writer_event.release()

        # Cam west
        videl_url_cam_west = DataTemplate.video_url(os.getenv('ID_CAM_WEST'), 'CAM_WEST')
        # Cam east
        videl_url_cam_east = DataTemplate.video_url(os.getenv('ID_CAM_EAST'), 'CAM_EAST')
        # Cam fish_eye
        id_cam_fish_eye = '_'.join(['ID', 'CAM_360'])
        videl_url_cam_fish_eye = DataTemplate.video_url(os.getenv(id_cam_fish_eye), 'CAM_360')
        # Cam shelf east
        if in_shelf == 1: cam_type = 'CAM_SHELF_01'
        elif in_shelf == 2: cam_type = 'CAM_SHELF_02'
        else: cam_type = None
        if cam_type is not None:
            id_cam_cam_shelf_east = '_'.join(['ID', cam_type, 'EAST'])
            id_cam_cam_shelf_mid = '_'.join(['ID', cam_type, 'MID'])
            id_cam_cam_shelf_west = '_'.join(['ID', cam_type, 'WEST'])
            videl_url_cam_shelf_east = DataTemplate.video_url(os.getenv(id_cam_cam_shelf_east),
                                                              '_'.join([cam_type, 'EAST']))
            videl_url_cam_shelf_mid = DataTemplate.video_url(os.getenv(id_cam_cam_shelf_mid),
                                                             '_'.join([cam_type, 'MID']))
            videl_url_cam_shelf_west = DataTemplate.video_url(os.getenv(id_cam_cam_shelf_west),
                                                              '_'.join([cam_type, 'WEST']))
            video_url_evidence = vms.get_video_url(
                [videl_url_cam_west, videl_url_cam_east, videl_url_cam_shelf_east, videl_url_cam_shelf_mid,
                 videl_url_cam_shelf_west, videl_url_cam_fish_eye], cur_time_in_ms - 7000,
                cur_time_in_ms + 3000)
        else:
            video_url_evidence = vms.get_video_url(
                [videl_url_cam_west, videl_url_cam_east, videl_url_cam_fish_eye], cur_time_in_ms - 7000, cur_time_in_ms + 3000)

        video_url_engine = 'http://{}:{}/{}'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'), video_name)
        data = DataTemplate.item_event(localIDs_item, cur_time, video_url_evidence, video_url_engine)
        event.save_event(data)

def rotate_image(img, angle=90, scale=1.0):
    h, w, c = img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_rotated = cv2.warpAffine(img, M, (w, h))
    return img_rotated
