import time
import cv2
import ast
import numpy as np
from shapely.geometry.polygon import Polygon
# from modules.Detection.Detector_ssd import PersonDetector
from modules.Detection.Detector_blitznet import PersonDetector
# from modules.Detection.Detector_yolov3 import PersonDetector
from modules.Tracking import Tracker
from modules.EventDetection import EventDetector
from modules.Visualization import Visualizer
from helpers.settings import *
from helpers.time_utils import get_timestamp_from_filename, convert_timestamp_to_human_time
from helpers.common_utils import CSV_Writer, draw_polygon, load_csv, map_id_shelf, map_id_signage, load_csv_signage

def process_cam_360(cam360_queue, num_loaded_model, global_tracks):

    engine_logger.critical('------ CAM_360 Engine process started ------')

    # Config parameters
    in_door_area = Polygon(ast.literal_eval(os.getenv('IN_DOOR_AREA')))
    out_door_area = Polygon(ast.literal_eval(os.getenv('OUT_DOOR_AREA')))
    shelf_a_area = Polygon(ast.literal_eval(os.getenv('A_AREA')))
    shelf_b_area = Polygon(ast.literal_eval(os.getenv('B_AREA')))
    a_left = Polygon(ast.literal_eval(os.getenv('A_LEFT')))
    a_right = Polygon(ast.literal_eval(os.getenv('A_RIGHT')))
    signage1_area = Polygon(ast.literal_eval(os.getenv('SIGNAGE1_AREA')))
    signage2_area = Polygon(ast.literal_eval(os.getenv('SIGNAGE2_AREA')))
    none_area = Polygon(ast.literal_eval(os.getenv('NONE_AREA')))
    roi_x1y1, roi_x2y2 = ast.literal_eval(os.getenv('ROI_CAM_360'))[0], ast.literal_eval(os.getenv('ROI_CAM_360'))[1]
    img_size_cam_360 = ast.literal_eval(os.getenv('IMG_SIZE_CAM_360'))
    shelf_ids_xy = ast.literal_eval(os.getenv('SHELF_IDS_XY'))

    # Create video writer
    if os.getenv('SAVE_VID') == 'TRUE':
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        vid_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), 'CAM_360.mp4')
        videoWriter = cv2.VideoWriter(vid_path, fourcc, int(os.getenv('FPS_CAM_360')), (512, 512))

    # Create list to store data in csv
    column_name = ['camera ID', 'shopper ID', 'process ID', 'info', 'timestamp (unix timestamp)',
                   'timestamp (UTC - JST)']
    csv_writer = CSV_Writer(column_name, os.getenv('CSV_CAM_360'))

    # Create instance of Visualizer
    visualizer = Visualizer(int(os.getenv('TRAJECTORIES_QUEUE_SIZE')))
    is_show = os.getenv('SHOW_GUI_360') == 'TRUE'

    # Create instance of PersonDetector
    # detector = PersonDetector(os.getenv('CAM_360_GPU'), os.getenv('YOLOv3_CFG_PATH'), ckpt_path=os.getenv('YOLOv3_MODEL_PATH'),
    #                           cls_names=os.getenv('CLS_PATH'), augment=False)
    # Or use Blitznet detection
    detector = PersonDetector(os.getenv('CAM_360_GPU'), os.getenv('CLS_BLITZNET_PATH'), os.getenv('BLITZNET_MODEL_PATH'))

    # Use SSD detection
    # detector = PersonDetector()

    detector.setROI(roi_x1y1, roi_x2y2)

    # Create instance of Tracker
    tracker = Tracker(int(os.getenv('MAX_AGE')), int(os.getenv('MIN_HITS')), float(os.getenv('LOW_IOU_THRESHOLD')),
                      float(os.getenv('MIN_DIST_PPL')), int(os.getenv('MIN_FREQ_PPL')))

    # Create instance of EventDetector
    event_detector = EventDetector(int(os.getenv('MIN_BASKET_FREQ')))

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
    # Load CSV shelf touch to combine
    csv_touch_path = 'log_shelf_touch.csv'
    csv_shelf_touch = load_csv(csv_touch_path)
    csv_shelf_touch['shopper ID'] = None
    index_touch = 0
    while (index_touch < len(csv_shelf_touch)) and (csv_shelf_touch['timestamp'][index_touch] <= start_timestamp):
        index_touch += 1
    # Load CSV signage to combine
    csv_signage1_path = 'log_signage_attention_01.csv'
    csv_signage1 = load_csv_signage(csv_signage1_path)
    csv_signage1['shopper ID'] = None
    index_signage1 = 0
    while (index_signage1 < len(csv_signage1)) and (csv_signage1['timestamp'][index_signage1] <= start_timestamp):
        index_signage1 += 1

    csv_signage2_path = 'log_signage_attention_02.csv'
    csv_signage2 = load_csv_signage(csv_signage2_path)
    csv_signage2['shopper ID'] = None
    index_signage2 = 0
    wait_frames = 0

    while (index_signage2 < len(csv_signage2)) and (csv_signage2['timestamp'][index_signage2] <= start_timestamp):
        index_signage2 += 1

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
        trackers, localIDs_end = tracker.update(dets, basket_dets, in_door_area, out_door_area, shelf_a_area, shelf_b_area, none_area)

        # Update event detection results
        event_detector.update(trackers, localIDs_end, csv_writer)

        # Combine touch CSV
        if (index_touch < len(csv_shelf_touch)) and (csv_shelf_touch['timestamp'][index_touch] <= cur_time):
            print('start map local_id to shelf touch')
            anchor_time = csv_shelf_touch['timestamp'][index_touch]
            list_shelf_id = [{'shelf_id': csv_shelf_touch['shelf ID'][index_touch],
                              'hand_xy': csv_shelf_touch['hand_coords'][index_touch],
                              'index': index_touch}]
            while (index_touch < len(csv_shelf_touch) - 1) and (csv_shelf_touch['shelf ID'][index_touch + 1] == anchor_time):
                index_touch += 1
                list_shelf_id.append({'shelf_id': csv_shelf_touch['shelf ID'][index_touch],
                                      'hand_xy': csv_shelf_touch['hand_coords'][index_touch],
                                      'index': index_touch})
            list_local_id = map_id_shelf(trackers, list_shelf_id, a_left, a_right, shelf_a_area, shelf_ids_xy)
            if (wait_frames > int(os.getenv('WAIT_FRAMES'))) or len(list_local_id) > 0:
                for shelf_info in list_shelf_id:
                    if (isinstance(shelf_info['local_id'], list)) and (len(shelf_info['local_id']) == 1): shelf_info['local_id'] = shelf_info['local_id'][0]
                    if (isinstance(shelf_info['local_id'], list)) and (len(shelf_info['local_id']) == 0): shelf_info['local_id'] = None
                    csv_shelf_touch['shopper ID'][shelf_info['index']] = shelf_info['local_id']
                    csv_writer.write((1, str(shelf_info['local_id']), 1201, 'SHELF ID {}'.format(shelf_info['shelf_id']),
                                      csv_shelf_touch['timestamp'][shelf_info['index']],
                                      csv_shelf_touch['timestamp (UTC - JST)'][shelf_info['index']]))
                index_touch += 1
                wait_frames = 0
            else: wait_frames += 1

        # Combine signage CSV
        if (index_signage1 < len(csv_signage1)) and (csv_signage1['timestamp'][index_signage1] <= cur_time):
            print('start map local_id to signage 1')
            local_id_signage = map_id_signage(trackers, signage1_area)
            time_has_attention = csv_signage1['Duration'][index_signage1].split(':')
            duration = float(time_has_attention[2]) + float(time_has_attention[3]) / 1000
            csv_signage1['Duration'][index_signage1] = '{}s'.format(duration)
            if (isinstance(local_id_signage, list)) and (len(local_id_signage) == 1): local_id_signage = local_id_signage[0]
            if (isinstance(local_id_signage, list)) and (len(local_id_signage) == 0): local_id_signage = None
            csv_signage1['shopper ID'][index_signage1] = str(local_id_signage)
            csv_writer.write((2, str(local_id_signage), 1517, 'HAS ATTENTION TO SIGNAGE1 IN {}s'.format(duration),
                              csv_signage1['timestamp'][index_signage1],
                              csv_signage1['Timestamp (UTC-JST)'][index_signage1]))
            index_signage1 += 1

        if (index_signage2 < len(csv_signage2)) and (csv_signage2['timestamp'][index_signage2] <= cur_time):
            print('start map local_id to signage 2')
            local_id_signage = map_id_signage(trackers, signage2_area)
            time_has_attention = csv_signage2['Duration'][index_signage2].split(':')
            duration = float(time_has_attention[2]) + float(time_has_attention[3]) / 1000
            csv_signage2['Duration'][index_signage2] = '{}s'.format(duration)
            if (isinstance(local_id_signage, list)) and (len(local_id_signage) == 1): local_id_signage = local_id_signage[0]
            if (isinstance(local_id_signage, list)) and (len(local_id_signage) == 0): local_id_signage = None
            csv_signage2['shopper ID'][index_signage2] = str(local_id_signage)
            csv_writer.write((2, str(local_id_signage), 1517, 'HAS ATTENTION TO SIGNAGE2 IN {}s'.format(duration),
                              csv_signage2['timestamp'][index_signage2],
                              csv_signage2['Timestamp (UTC-JST)'][index_signage2]))
            index_signage2 += 1


        # Visualization: plot bounding boxes & trajectories
        # draw_polygon(img_ori, ast.literal_eval(os.getenv('SIGNAGE2_AREA')))
        # draw_polygon(img_ori, ast.literal_eval(os.getenv('SIGNAGE1_AREA')))
        # draw_polygon(img_ori, ast.literal_eval(os.getenv('OUT_DOOR_AREA')))
        visualizer.draw(img_ori, basket_dets, trackers, event_detector)

        # Display the resulting frame
        # cv2.putText(img_ori, 'Frame #{:d} ({:.2f}ms)'.format(frame_cnt, (time.time() - start_time) * 1000), (2, 35),
        #             0, fontScale=0.6, color=(0, 255, 0), thickness=2)
        # cv2.putText(img_ori, '{}'.format(convert_timestamp_to_human_time(cur_time)), (2, 305),
        #             0, fontScale=0.6, color=(0, 255, 0), thickness=2)
        if is_show: visualizer.show(img=img_ori, title='Fish-eye camera Detection and Tracking Python Demo')


        if len(event_detector.localIDs_entered) > 0 or len(event_detector.localIDs_exited) > 0 or len(event_detector.localIDs_A) > 0 or len(event_detector.localIDs_B) > 0:
            for i in range(0, 10): videoWriter.write(img_ori)

        engine_logger.info('Frame Count: {} - CAM_360 Flow with FPS: {}'.format(frame_cnt, 1. / (time.time() - start_time)))

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
        if trk.basket_count > int(os.getenv('MIN_BASKET_FREQ')):
            csv_writer.write((1, trk.id, 1202, 'HAS BASKET', int(trk.basket_time),
                        convert_timestamp_to_human_time(int(trk.basket_time))))
        else:
            csv_writer.write((1, trk.id, 1202, 'NO BASKET', int(trk.basket_time),
                        convert_timestamp_to_human_time(int(trk.basket_time))))
        ppl_accompany = np.asarray(list(trk.ppl_dist.values()))
        ppl_accompany = ppl_accompany[ppl_accompany > int(os.getenv('MIN_FREQ_PPL'))]
        csv_writer.write((1, trk.id, 1203, 'GROUP {} PEOPLE'.format(len(ppl_accompany) + 1), int(cur_time), convert_timestamp_to_human_time(int(cur_time))))

    csv_writer.to_csv(sep=',', index_label='ID', sort_column=['shopper ID', 'timestamp (unix timestamp)'])

    csv_shelf_touch.to_csv(os.getenv('CSV_CAM_SHELF'), index=False)
    csv_signage1.to_csv(os.getenv('CSV_CAM_SIGNAGE_01'), index=False)
    csv_signage2.to_csv(os.getenv('CSV_CAM_SIGNAGE_02'), index=False)

    engine_logger.info('Created successfully CSV file of CAM_360 !')

    engine_logger.critical('------ CAM_360 Engine process stopped ------')