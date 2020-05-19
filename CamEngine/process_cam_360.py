import time
import cv2
import ast
import numpy as np
from collections import deque
from shapely.geometry.polygon import Polygon
from modules.Detection.Detector_yolov3 import PersonDetector
from modules.Tracking import Tracker
from helpers.settings import *
from helpers.time_utils import get_timestamp_from_filename, convert_to_human_time
from helpers.common_utils import plot_bbox, plot_tracjectories
import pandas as pd

def process_cam_360(cam360_queue, num_loaded_model, global_tracks):

    engine_logger.critical('------ CAM_360 Engine process started ------')

    # Config parameters
    in_door_area = Polygon(ast.literal_eval(os.getenv('IN_DOOR_AREA')))
    out_door_area = Polygon(ast.literal_eval(os.getenv('OUT_DOOR_AREA')))
    shelf_a_area = Polygon(ast.literal_eval(os.getenv('A_AREA')))
    shelf_b_area = Polygon(ast.literal_eval(os.getenv('B_AREA')))
    none_area = Polygon(ast.literal_eval(os.getenv('NONE_AREA')))
    roi_x1y1, roi_x2y2 = ast.literal_eval(os.getenv('ROI_CAM_360'))[0], ast.literal_eval(os.getenv('ROI_CAM_360'))[1]
    img_size_cam_360 = ast.literal_eval(os.getenv('IMG_SIZE_CAM_360'))

    # Write video
    if os.getenv('SAVE_VID') == 'TRUE':
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        vid_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), 'CAM_360.mp4')
        videoWriter = cv2.VideoWriter(vid_path, fourcc, int(os.getenv('FPS_CAM_360')), img_size_cam_360)

    # Create list to store data in csv
    csv_data = []
    PROCESS_ID = int(os.getenv('PROCESS_ID_360'))
    CAMERA_ID = int(os.getenv('CAMERA_ID'))

    # buffer is the maximum size of plot deque
    pts = deque(maxlen=int(os.getenv('TRAJECTORIES_QUEUE_SIZE')))

    # Create color table
    np.random.seed(10)
    colours = np.random.randint(0, 256, size=(16, 3))

    # Create instance of PersonDetector
    detector = PersonDetector(os.getenv('CAM_360_GPU'), os.getenv('CFG_PATH'), ckpt_path=os.getenv('YOLOv3_MODEL_PATH'),
                              cls_names=os.getenv('CLS_PATH'), augment=False)
    detector.setROI(roi_x1y1, roi_x2y2)

    # Create instance of Tracker
    tracker = Tracker(int(os.getenv('MAX_AGE')), int(os.getenv('MIN_HITS')), float(os.getenv('LOW_IOU_THRESHOLD')))

    # Start frameID
    frame_cnt = 0

    # Initial check event
    num_model = len(os.getenv('LIST_CAM_ENGINE').split(','))
    num_loaded_model.value += 1

    # Start AI engine
    engine_logger.critical('Tracking engine has local_id start from {}'.format(1))
    vid = cv2.VideoCapture(os.getenv('RTSP_CAM_360'))
    try:
        start_timestamp = get_timestamp_from_filename(os.path.basename(os.getenv('RTSP_CAM_360')))
    except:
        start_timestamp = time.time()

    while vid.isOpened():
        # if (num_loaded_model.value < num_model) or (cam360_queue.qsize() <= 0):
        #    continue
        # img_data = cam360_queue.get()
        # img1 = img_data['frame']
        _, img1 = vid.read()
        if not _: break
        img_ori = cv2.resize(img1, img_size_cam_360)
        cur_time = start_timestamp + vid.get(cv2.CAP_PROP_POS_MSEC) / 1000
        start_time = time.time()

        # Get Detection results
        detector.setFrame(img_ori[roi_x1y1[1]:roi_x2y2[1], roi_x1y1[0]:roi_x2y2[0]])
        dets, basket_dets = detector.getOutput()

        # Get Tracking result
        tracker.setTimestamp(cur_time)
        trackers, trks_start, localIDs_end = tracker.update(dets, in_door_area, out_door_area, shelf_a_area, shelf_b_area, none_area)

        # Share trackers to shelf cam process
        # global_tracks.to_redis(trackers, int(img_data['timestamp'] * 1000))

        # Check ENTER event
        localIDs_entered = []

        # Check EXIT event
        localIDs_exited = []

        # Visualization
        centers = []
        localIDs_A = []
        localIDs_B = []
        plot_bbox(img_ori, basket_dets, colours=colours)
        plot_bbox(img_ori, trackers, colours=colours)

        for d in trackers:
            if d[-1] > 0:
                # Check bit area
                bit_area = int(d[-3])
                jp_time = convert_to_human_time(d[-4])
                if bit_area == 1:
                    localIDs_entered.append(int(d[-1]))
                    csv_data.append((CAMERA_ID, int(d[-1]), PROCESS_ID, 'ENTRANCE', int(d[-4]), jp_time))
                if bit_area == 2:
                    localIDs_exited.append(int(d[-1]))
                    csv_data.append((CAMERA_ID, int(d[-1]), PROCESS_ID, 'EXIT', int(d[-4]), jp_time))
                if bit_area == 3:
                    csv_data.append((CAMERA_ID, int(d[-1]), PROCESS_ID, 'A', int(d[-4]), jp_time))
                    localIDs_A.append(int(d[-1]))
                if bit_area == 4:
                    csv_data.append((CAMERA_ID, int(d[-1]), PROCESS_ID, 'B', int(d[-4]), jp_time))
                    localIDs_B.append(int(d[-1]))

                centers.append((int((d[0] + d[2]) / 2), int((d[1] + d[3]) / 2), int(d[-1])))

        # Plot trajectories
        pts.appendleft(centers)
        plot_tracjectories(img_ori, pts, colours=colours)

        # Display the resulting frame
        cv2.putText(img_ori, 'Frame #{:d} ({:.2f}ms)'.format(frame_cnt, (time.time() - start_time) * 1000), (2, 35),
                    0, fontScale=0.6, color=(0, 255, 0), thickness=2)

        if len(localIDs_entered) > 0:
            cv2.putText(img_ori, 'ENTER:{}'.format(", ".join(map(str, localIDs_entered))), (2, 55), 0, fontScale=0.6,
                        color=(0, 0, 255), thickness=2)
        if len(localIDs_exited) > 0:
            cv2.putText(img_ori, 'EXIT:{}'.format(", ".join(map(str, localIDs_exited))), (2, 75), 0, fontScale=0.6,
                        color=(0, 0, 255), thickness=2)
        if len(localIDs_A) > 0:
            cv2.putText(img_ori, 'A:{}'.format(", ".join(map(str, localIDs_A))), (2, 95), 0, fontScale=0.6,
                        color=(0, 0, 255), thickness=2)
        if len(localIDs_B) > 0:
            cv2.putText(img_ori, 'B:{}'.format(", ".join(map(str, localIDs_B))), (2, 105), 0, fontScale=0.6,
                        color=(0, 0, 255), thickness=2)

        if len(localIDs_entered) > 0 or len(localIDs_exited) > 0 or len(localIDs_A) > 0 or len(localIDs_B) > 0:
            for i in range(0, 10): videoWriter.write(img_ori)

        engine_logger.info('Frame Count: {} - CAM_360 Flow with FPS: {}'.format(frame_cnt, 1. / (time.time() - start_time)))

        if os.getenv('SHOW_GUI_360') == 'TRUE':
            cv2.imshow('Fish-eye camera Detection and Tracking Python Demo', img_ori)
            cv2.waitKey(1)

        if os.getenv('SAVE_VID') == 'TRUE':
            videoWriter.write(img_ori)

        frame_cnt += 1

    # Create result csv file
    column_name = ['camera ID', 'shopper ID', 'process ID', 'area', 'timestamp (unix timestamp)', 'timestamp (UTC - JST)']
    csv_df = pd.DataFrame(csv_data, columns=column_name)
    csv_df.to_csv(os.getenv('CSV_CAM_360'), index=True, index_label='ID')
    engine_logger.info('Created successfully CSV file of CAM_360 !')

    # Closes all the frames
    if os.getenv('SAVE_VID') == 'TRUE':
        videoWriter.release()

    cv2.destroyAllWindows()
    engine_logger.critical('------ CAM_360 Engine process stopped ------')