import time
import cv2
import ast
import numpy as np
from shapely.geometry.polygon import Polygon
from modules.Detection.Detector_yolov5 import PersonFaceDetector
from modules.Tracking import SignageTracker
from modules.EventDetection import EventDetector
from modules.Visualization import Visualizer
from helpers.settings import *
from helpers.time_utils import get_timestamp_from_filename, convert_timestamp_to_human_time
from helpers.common_utils import CSV_Writer, draw_polygon, post_processing_signage_csv, calculate_duration, set_camera_id
from modules.Headpose.Detector_headpose import HeadposeDetector


def process_cam_signage(cam_signage_queue, num_loaded_model):

    # Config parameters
    roi_x1y1, roi_x2y2 = ast.literal_eval(os.getenv('ROI_CAM_SIGNAGE'))[0], \
                         ast.literal_eval(os.getenv('ROI_CAM_SIGNAGE'))[1]
    img_size_cam_signage = ast.literal_eval(os.getenv('IMG_SIZE_CAM_SIGNAGE'))
    signage1_enter_area = Polygon(ast.literal_eval(os.getenv('SIGNAGE1_ENTER_AREA')))

    # Get cam signage id
    for i in range(1, 3):
    # Get cam signage id
        engine_logger.critical('------ CAM_SIGNAGE_{:02} Engine process started ------'.format(i))
        cam_id = set_camera_id(i)
        # Create video writer
        if os.getenv('SAVE_VID') == 'TRUE':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            vid_path = join(os.getenv('OUTPUT_DIR'), 'CAM_SIGNAGE_{:02}.mp4'.format(cam_id))
            videoWriter = cv2.VideoWriter(vid_path, fourcc, int(os.getenv('FPS_CAM_SIGNAGE')), img_size_cam_signage)

        # Create list to store data in csv
        column_name = ['camera ID', 'shopper ID', 'process ID', 'info', 'Start_time',
                       'End_time', 'Duration(s)', 'Start_bbox_x', 'Start_bbox_y', 'End_bbox_x',
                       'End_bbox_y']
        csv_writer = CSV_Writer(column_name, os.getenv('CSV_CAM_SIGNAGE_{:02}'.format(cam_id)))

        # Create instance of Visualizer
        visualizer = Visualizer(int(os.getenv('TRAJECTORIES_QUEUE_SIZE')))
        is_show = os.getenv('SHOW_GUI_360') == 'TRUE'

        # Create instance of PersonDetector
        detector = PersonFaceDetector(os.getenv('CAM_360_GPU'), os.getenv('YOLOv5_SIGNAGE_CFG_PATH'),
                                      ckpt_path=os.getenv('YOLOv5_SIGNAGE_MODEL_PATH'), augment=False)
        detector.setROI(roi_x1y1, roi_x2y2)

        # Create instance of HeadposeDetector
        hpDetector = HeadposeDetector(os.getenv('HEADPOSE_MODEL_PATH'))

        # Create instance of Tracker
        tracker = SignageTracker(int(os.getenv('MAX_AGE')), int(os.getenv('MIN_HITS')),
                                 float(os.getenv('LOW_IOU_THRESHOLD')), float(os.getenv('MIN_AREA_RATIO')),
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
        vid = cv2.VideoCapture(os.getenv('RTSP_CAM_SIGNAGE_{:02}'.format(i)))
        try:
            start_timestamp = get_timestamp_from_filename(os.path.basename(os.getenv('RTSP_CAM_SIGNAGE_{:02}'.format(i))))
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
            trackers, localIDs_end = tracker.update(dets, faces, hpDetector, img_ori)

            # Update event detection results
            event_detector.update_signage(localIDs_end, csv_writer)

            visualizer.draw_signage(img_ori, faces, trackers)

            draw_polygon(img_ori, ast.literal_eval(os.getenv('SIGNAGE1_ENTER_AREA')))

            # Display the resulting frame
            cv2.putText(img_ori, 'Frame #{:d} ({:.2f}ms)'.format(frame_cnt, (time.time() - start_time) * 1000), (2, 35),
                        0, fontScale=0.6, color=(0, 255, 0), thickness=2)
            if is_show: visualizer.show(img=img_ori, title='Fish-eye camera Detection and Tracking Python Demo')

            engine_logger.info(
                'Frame Count: {} - CAM_Signage Flow with FPS: {}'.format(frame_cnt, 1. / (time.time() - start_time)))

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
            duration_group = calculate_duration(trk.basket_time, cur_time)
            x2_center = int((min(max(trk.get_state()[0][0], 0), img_size_cam_signage[0]) + min(
                max(trk.get_state()[0][2], 0), img_size_cam_signage[0])) / 2)
            y2_center = int((min(max(trk.get_state()[0][1], 0), img_size_cam_signage[1]) + min(
                max(trk.get_state()[0][3], 0), img_size_cam_signage[1])) / 2)
            csv_writer.write((cam_id, trk.id, 1203, 'GROUP {} PEOPLE'.format(len(ppl_accompany) + 1),
                              convert_timestamp_to_human_time(trk.basket_time), convert_timestamp_to_human_time(cur_time),
                              duration_group,
                              int((trk.sig_start_bbox[0] + trk.sig_start_bbox[2]) / 2),
                              int((trk.sig_start_bbox[1] + trk.sig_start_bbox[3]) / 2),
                              x2_center, y2_center))

            if len(trk.duration_hp_list) != 0:
                for start, end, duration in zip(trk.start_hp_list, trk.end_hp_list, trk.duration_hp_list):
                    csv_writer.write((cam_id, trk.id, 1557, 'has_attention', convert_timestamp_to_human_time(start),
                                      convert_timestamp_to_human_time(end), duration, None, None, None, None))

        csv_writer.to_csv(sep=',', index_label='ID', sort_column=['shopper ID'])

        # Post Processing Camera Signage
        post_processing_signage_csv(input_csv=os.getenv('CSV_CAM_SIGNAGE_{:02}'.format(cam_id)),
                                    output_csv=os.getenv('PROCESSED_CSV_SIGNAGE_{:02}_PATH'.format(cam_id)),
                                    signage_enter_area=signage1_enter_area)
        engine_logger.info('Created successfully CSV file of CAM_Signage {:02}!'.format(cam_id))
        engine_logger.critical('------ CAM_SIGNAGE_{:02} Engine process stopped ------'.format(i))
