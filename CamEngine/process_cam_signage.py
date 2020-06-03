import time
import cv2
import ast
import numpy as np
from shapely.geometry.polygon import Polygon
# from modules.Detection.Detector_ssd import PersonDetector
# from modules.Detection.Detector_blitznet import PersonDetector
from modules.Detection.Detector_yolov3 import PersonDetector
from modules.Tracking import SignageTracker
from modules.EventDetection import EventDetector
from modules.Visualization import Visualizer
from helpers.settings import *
from helpers.time_utils import get_timestamp_from_filename, convert_timestamp_to_human_time
from helpers.common_utils import CSV_Writer, draw_polygon, load_csv, map_id_shelf,calculate_duration
from modules.Headpose.Detector_headpose import HeadposeDetector


def process_cam_signage(cam_signage_queue, num_loaded_model):

    engine_logger.critical('------ CAM_SIGNAGE Engine process started ------')

    # Config parameters
    roi_x1y1, roi_x2y2 = ast.literal_eval(os.getenv('ROI_CAM_SIGNAGE'))[0], ast.literal_eval(os.getenv('ROI_CAM_SIGNAGE'))[1]
    img_size_cam_signage = ast.literal_eval(os.getenv('IMG_SIZE_CAM_SIGNAGE'))

    # Create video writer
    if os.getenv('SAVE_VID') == 'TRUE':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_path = join(os.getenv('OUTPUT_DIR'), 'CAM_SIGNAGE.mp4')
        videoWriter = cv2.VideoWriter(vid_path, fourcc, int(os.getenv('FPS_CAM_360')), img_size_cam_signage)

    # Create list to store data in csv
    column_name = ['camera ID', 'shopper ID', 'process ID', 'info', 'Start_time',
                   'End_time','Duration(s)']
    csv_writer = CSV_Writer(column_name, os.getenv('CSV_CAM_SIGNAGE_01'))

    # Create instance of Visualizer
    visualizer = Visualizer(int(os.getenv('TRAJECTORIES_QUEUE_SIZE')))
    is_show = os.getenv('SHOW_GUI_360') == 'TRUE'

    # Create instance of PersonDetector
    detector = PersonDetector(os.getenv('CAM_360_GPU'), os.getenv('YOLOv3_SIGNAGE_CFG_PATH'), ckpt_path=os.getenv('YOLOv3_SIGNAGE_MODEL_PATH'),
                              cls_names=os.getenv('CLS_SIGNAGE_PATH'), augment=False)

    detector.setROI(roi_x1y1, roi_x2y2)


    # Create instance of HeadposeDetector
    hpDetector = HeadposeDetector(os.getenv('HEADPOSE_MODEL_PATH'))

    # Create instance of Tracker
    tracker = SignageTracker(int(os.getenv('MAX_AGE')), int(os.getenv('MIN_HITS')), float(os.getenv('LOW_IOU_THRESHOLD')), float(os.getenv('MIN_AREA_RATIO')), 
                                float(os.getenv('MAX_AREA_RATIO')), int(os.getenv('MIN_AREA_FREQ')))

    # Create instance of EventDetector
    event_detector = EventDetector(int(os.getenv('MIN_BASKET_FREQ')))

    # Start frameID
    frame_cnt = 0

    # Initial check event
    num_model = len(os.getenv('LIST_CAM_ENGINE').split(','))
    num_loaded_model.value += 1

    # Start AI engine
    engine_logger.critical('Tracking engine has local_id start from {}'.format(1))
    vid = cv2.VideoCapture(os.getenv('RTSP_CAM_SIGNAGE'))
    try:
        start_timestamp = get_timestamp_from_filename(os.path.basename(os.getenv('RTSP_CAM_SIGNAGE')))
    except:
        start_timestamp = time.time()

    while vid.isOpened():
        _, img1 = vid.read()
        if not _: break
        img_ori = cv2.resize(img1, img_size_cam_signage)
        cur_time = start_timestamp + vid.get(cv2.CAP_PROP_POS_MSEC) / 1000
        start_time = time.time()

        # Get Detection results
        detector.setFrame(img_ori[roi_x1y1[1]:roi_x2y2[1], roi_x1y1[0]:roi_x2y2[0]])
        dets, faces = detector.getOutput()

        # Get Tracking result
        tracker.setTimestamp(cur_time)

        # local_IDs_end: dead tracklets , clean up the trackers 
        trackers, localIDs_end = tracker.update(dets, faces,hpDetector,img_ori)

        # Update event detection results
        event_detector.update_signage(trackers, faces, localIDs_end, csv_writer)

        visualizer.draw_signage(img_ori, faces, trackers, event_detector)

        # Display the resulting frame
        cv2.putText(img_ori, 'Frame #{:d} ({:.2f}ms)'.format(frame_cnt, (time.time() - start_time) * 1000), (2, 35),
                    0, fontScale=0.6, color=(0, 255, 0), thickness=2)
        if is_show: visualizer.show(img=img_ori, title='Fish-eye camera Detection and Tracking Python Demo')

        engine_logger.info('Frame Count: {} - CAM_Signage Flow with FPS: {}'.format(frame_cnt, 1. / (time.time() - start_time)))

        if os.getenv('SAVE_VID') == 'TRUE':
            videoWriter.write(img_ori)

        frame_cnt += 1

    # Closes all the frames
    if os.getenv('SAVE_VID') == 'TRUE':
        videoWriter.release()
    cv2.destroyAllWindows()

    # Add remaining and existing track to csv_data if end video
    for trk in tracker._trackers:
        if trk.id < 0: continue
        ppl_accompany = np.asarray(list(trk.ppl_dist.values()))
        ppl_accompany = ppl_accompany[ppl_accompany > int(os.getenv('MIN_AREA_FREQ'))]
        # change to new format
        duration_group = calculate_duration(trk.basket_time,int(cur_time))
        csv_writer.write((1, trk.id, 1203, 'GROUP {} PEOPLE'.format(len(ppl_accompany) + 1), 
                                    convert_timestamp_to_human_time(trk.basket_time), 
                                    convert_timestamp_to_human_time(int(cur_time)),
                                    duration_group))

        if trk.cnt_frame_attention > int(os.getenv('THRESHOLD_HEADPOSE')):
            csv_writer.write((1,trk.id,1557,'has_attention',convert_timestamp_to_human_time(trk.start_hp_time),
                                                            convert_timestamp_to_human_time(trk.end_hp_time),'{}'.format(str((trk.cnt_frame_attention)/ int(os.getenv('FPS_CAM_SIGNAGE'))))))
    
    csv_writer.to_csv(sep=',', index_label='ID', sort_column=['shopper ID'])

    engine_logger.info('Created successfully CSV file of CAM_Signage !')

    engine_logger.critical('------ CAM_Signage Engine process stopped ------') 