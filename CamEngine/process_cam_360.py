import time
import cv2
import ast
import redis
import uuid
import numpy as np
from collections import deque
from shapely.geometry.polygon import Polygon
from modules.DataTemplate import DataTemplate
# from modules.Detection.Detector_ssd import PersonDetector
# from modules.Detection.Detector_blitznet import Bliznet_detector as PersonDetector
from modules.Detection.Detector_yolov3 import PersonDetector
from modules.EventManager import EventManager
from modules.Tracking import Tracker
from modules.VMSManager import VMSManager
from helpers.settings import *
from helpers.concat_local_id import concat_local_id_time, convert_to_jp_time
from modules.Tracking.utils.check_overwrap import check_overwrap
import pandas as pd

def process_cam_360(cam360_queue, num_loaded_model, global_tracks):

    engine_logger.critical('------ CAM_360 Engine process started ------')

    # Write video
    if os.getenv('SAVE_VID') == 'TRUE':
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        vid_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), 'CAM_360.mp4')
        videoWriter = cv2.VideoWriter(vid_path, fourcc, int(os.getenv('FPS_CAM_360')), (512, 512))

    # Create list to store data in csv
    csv_data = []
    PROCESS_ID = int(os.getenv('PROCESS_ID_360'))
    CAMERA_ID = int(os.getenv('CAMERA_ID'))

    # buffer is the maximum size of plot deque
    pts = deque(maxlen=int(os.getenv('TRAJECTORIES_QUEUE_SIZE')))

    # Create color table
    np.random.seed(10)
    colours = np.random.randint(0, 256, size=(16, 3))

    # Config parameters
    IN_DOOR_AREA = list(ast.literal_eval(os.getenv('IN_DOOR_AREA')))
    OUT_DOOR_AREA = list(ast.literal_eval(os.getenv('OUT_DOOR_AREA')))
    A_AREA = list(ast.literal_eval(os.getenv('A_AREA')))
    B_AREA = list(ast.literal_eval(os.getenv('B_AREA')))
    NONE_AREA = list(ast.literal_eval(os.getenv('NONE_AREA')))
    in_door_box = Polygon(IN_DOOR_AREA)
    out_door_box = Polygon(OUT_DOOR_AREA)
    a_box = Polygon(A_AREA)
    b_box = Polygon(B_AREA)
    none_box = Polygon(NONE_AREA)
    roi_x1y1 = list(ast.literal_eval(os.getenv('ROI_CAM_360')))[0]
    roi_x2y2 = list(ast.literal_eval(os.getenv('ROI_CAM_360')))[1]

    # Create instance of Redis
    r = redis.Redis(os.getenv("REDIS_HOST"), int(os.getenv("REDIS_PORT")))

    # Create instance of Event Manager
    # event = EventManager()
    # event.send_to_bpo()

    # Create instance of VMS Manager
    # vms = VMSManager()

    # Create instance of PersonDetector
    detector = PersonDetector(os.getenv('CAM_360_GPU'), os.getenv('CFG_PATH'), ckpt_path=os.getenv('YOLO_MODEL_PATH'),
                              cls_names=os.getenv('CLS_PATH'), augment=False)
    detector.setROI(roi_x1y1, roi_x2y2)

    # Create instance of Tracker
    trackers = []
    tracker = Tracker(int(os.getenv('MAX_AGE')), int(os.getenv('MIN_HITS')), float(os.getenv('LOW_IOU_THRESHOLD')))

    # Start frameID
    frame_cnt = 0

    # Initial check event
    # engine_frames = deque(maxlen=20)
    num_model = len(os.getenv('LIST_CAM_ENGINE').split(','))
    num_loaded_model.value += 1

    # Get latest local_id
    # last_local_id = event.find_latest_local_id()
    # if last_local_id is None: last_local_id = int(os.getenv('LAST_ID'))
    # else: last_local_id = int(last_local_id.split('_')[0])
    last_local_id = 0
    r.set(name='count_id', value=last_local_id)

    # Start AI engine
    engine_logger.critical('Tracking engine has local_id start from {}'.format(last_local_id + 1))
    vid = cv2.VideoCapture(os.getenv('RTSP_CAM_360'))

    while vid.isOpened():
        #if (num_loaded_model.value < num_model) or (cam360_queue.qsize() <= 0):
        #    continue
        _, img1 = vid.read()
        if not _: break
        img_data = dict()
        img_data['frame'] = cv2.resize(img1, (512, 512))
        img_data['timestamp'] = time.time()
        start_time = time.time()
        #img_data = cam360_queue.get()
        img_ori = img_data['frame']
        cur_time = int(img_data['timestamp'])

        # Get Detection results
        detector.setFrame(img_ori[roi_x1y1[1]:roi_x2y2[1], roi_x1y1[0]:roi_x2y2[0]], frame_cnt)
        dets = detector.getOutput()

        # Get Tracking result
        prev_trackers = trackers
        tracker.setFrameID(frame_cnt)
        tracker.setTimestamp(cur_time)
        trackers, trks_start, localIDs_end = tracker.update(dets, in_door_box, out_door_box, a_box, b_box, none_box)

        # Save or update persons to persons collection in db
        # save_persons_toDB(event, trks_start, localIDs_end, img_ori, cur_time)

        # Share trackers to shelf cam process
        # global_tracks.to_redis(trackers, int(img_data['timestamp'] * 1000))

        # Check OVERWRAP event
        overwrap_cases, overwrap_to_draw = check_overwrap(prev_trackers, trackers, float(os.getenv('OVERWRAP_IOU_THRESHOLD')))

        # Check ENTER event
        localIDs_entered = []

        # Check EXIT event
        localIDs_exited = []

        # Visualization
        # Plot bbox with trackID
        centers = []
        # cv2.polylines(img_ori, [np.asarray(IN_DOOR_AREA, np.int32)], True, (0, 0, 255), thickness=2)
        # cv2.polylines(img_ori, [np.asarray(OUT_DOOR_AREA, np.int32)], True, (0, 255, 0), thickness=2)
        # cv2.polylines(img_ori, [np.asarray(A_AREA, np.int32)], True, (0, 0, 255), thickness=2)
        # cv2.polylines(img_ori, [np.asarray(B_AREA, np.int32)], True, (0, 0, 255), thickness=2)
        localIDs_A = []
        localIDs_B = []
        for d in trackers:
            # Plot bndbox
            if d[-1] > 0:
                # Check bit area
                bit_area = int(d[-3])
                jp_time = convert_to_jp_time(d[-4])
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

                color = colours[int(d[-1]) % 16].tolist()
                tl = round(0.001 * (img_ori.shape[0] + img_ori.shape[1]) / 2) + 1  # line thickness
                c1, c2 = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
                cv2.rectangle(img_ori, c1, c2, color, thickness=tl)
                # Plot score
                tf = max(tl - 1, 1)  # font thickness
                label = '%d' % int(d[-1])
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img_ori, c1, c2, color, -1)  # filled
                cv2.putText(img_ori, label, (c1[0], c1[1] - 2), 0, tl / 4, [0, 0, 0], thickness=tf,
                            lineType=cv2.LINE_AA)
                centers.append((int((d[0] + d[2]) / 2), int((d[1] + d[3]) / 2), int(d[-1])))

        # Plot trajectories
        pts.appendleft(centers)
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore them
            if pts[i - 1] is None or pts[i] is None:
                continue
            for j in range(0, len(pts[i - 1])):
                for k in range(0, len(pts[i])):
                    if (pts[i-1][j][2] == pts[i][k][2]) and (pts[i-1][j][2] > 0):
                        color = colours[pts[i-1][j][2] % 16].tolist()
                        cv2.line(img_ori, pts[i-1][j][0:2], pts[i][k][0:2], color, thickness=2)
                        continue

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
        if len(overwrap_cases) > 0:
            overwrap_to_draw = np.asarray(overwrap_to_draw)
            cv2.putText(img_ori, 'OVERWRAP:{}'.format(", ".join(map(str, overwrap_to_draw))), (200, 30), 0, fontScale=0.7,
                        color=(0, 0, 255),
                        thickness=2)

        if len(localIDs_entered) > 0 or len(localIDs_exited) > 0 or len(localIDs_A) > 0 or len(localIDs_B) > 0:
            for i in range(0, 20): videoWriter.write(img_ori)

        # Save events to events collection in db
        # save_events_toDB(vms, event, localIDs_entered, localIDs_exited, overwrap_cases, cur_time, engine_frames, r)
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

def save_events_toDB(vms, event, localIDs_entered, localIDs_exited, overwrap_cases, cur_time, engine_frames, r):

    # Save ENTER event to database
    cur_time_in_ms = cur_time * 1000
    if len(localIDs_entered) > 0:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_name = '{}.mp4'.format(str(uuid.uuid1()))
        video_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), video_name)
        write_event = cv2.VideoWriter(video_path, fourcc, 5, (704, 576))
        for i in range(0, len(engine_frames)):
            write_event.write(engine_frames[i])
        write_event.release()

        # Cam entrance
        videl_url_cam_entrance = DataTemplate.video_url(os.getenv('ID_CAM_ENTRANCE'), 'CAM_ENTRANCE')
        video_url_engine = 'http://{}:{}/{}'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'), video_name)
        video_url_evidence = vms.get_video_url([videl_url_cam_entrance], cur_time_in_ms - 5000, cur_time_in_ms + 5000)
        data = DataTemplate.enter_event(localIDs_entered, cur_time, video_url_evidence, video_url_engine)
        event.save_event(data)


    # Save EXIT event to database
    if len(localIDs_exited) > 0:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_name = '{}.mp4'.format(str(uuid.uuid1()))
        video_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), video_name)
        write_event = cv2.VideoWriter(video_path, fourcc, 5, (704, 576))
        for i in range(0, len(engine_frames)):
            write_event.write(engine_frames[i])
        write_event.release()

        # Cam exit
        video_url_cam_exit = DataTemplate.video_url(os.getenv('ID_CAM_EXIT'), 'CAM_EXIT')
        video_url_engine = 'http://{}:{}/{}'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'), video_name)
        video_url_evidence = vms.get_video_url([video_url_cam_exit], cur_time_in_ms - 5000, cur_time_in_ms + 5000)
        data = DataTemplate.exit_event(localIDs_exited, cur_time, video_url_evidence, video_url_engine)
        event.save_event(data)

    # Save OVERWRAP event to database
    if len(overwrap_cases) > 0:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_name = '{}.mp4'.format(str(uuid.uuid1()))
        video_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), video_name)
        write_event = cv2.VideoWriter(video_path, fourcc, 5, (704, 576))
        for i in range(0, len(engine_frames)):
            write_event.write(engine_frames[i])
        write_event.release()

        # Cam west
        videl_url_cam_west = DataTemplate.video_url(os.getenv('ID_CAM_WEST'), 'CAM_WEST')
        # Cam east
        videl_url_cam_east = DataTemplate.video_url(os.getenv('ID_CAM_EAST'), 'CAM_EAST')
        # Cam fish-eye
        videl_url_cam_fish_eye = DataTemplate.video_url(os.getenv('ID_CAM_360'), 'CAM_360')

        video_url_evidence = vms.get_video_url([videl_url_cam_west, videl_url_cam_east, videl_url_cam_fish_eye], cur_time_in_ms - 5000, cur_time_in_ms + 5000)
        video_url_engine = 'http://{}:{}/{}'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'), video_name)
        for overwrap_case in overwrap_cases:
            query = {'event_type': 'OVERWRAP', 'local_id': {'$in': [[overwrap_case], [overwrap_case[::-1]]]}, 'timestamp': {'$gt': cur_time - int(os.getenv('OVERWRAP_OFFSET_TIME'))}}
            same_event = event.find_event(query)
            if len(same_event) > 0: continue
            data = DataTemplate.overwrap_event([overwrap_case], cur_time, video_url_evidence, video_url_engine)
            event.save_event(data)

def save_persons_toDB(event, trks_start, localIDs_end, img_ori, cur_time):

    # Insert new person to db
    for trk_start in trks_start:
        local_id_time = concat_local_id_time(trk_start[-1], trk_start[-2])
        cropped_img_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), '{}.jpg'.format(local_id_time))
        cropped_img = img_ori[int(trk_start[1]):int(trk_start[3]), int(trk_start[0]):int(trk_start[2])]
        cv2.imwrite(cropped_img_path, cropped_img)
        cropped_img_path = 'http://{}:{}/{}.jpg'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'),
                                                                local_id_time)
        person = DataTemplate.person_document(local_id_time, cropped_img_path, int(trk_start[-2]), None)
        event.save_person(person)

    # Update person to db
    for local_id in localIDs_end:
        query = {'local_id': local_id}
        event.update_person(query, {'exit_time': cur_time})
