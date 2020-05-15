import cv2
import uuid
import numpy as np
import pandas as pd
import ast
import redis
import time
from collections import deque
from helpers.settings import *
from modules.Detectron import Detectron
from modules.Recognition import Recognition
from modules.EventManager import EventManager
from modules.VMSManager import VMSManager
from modules.DataTemplate import DataTemplate
from helpers.concat_local_id import concat_local_id_time


def process_cam_sacker(cam_queue, num_loaded_model, global_tracks):
    engine_logger.critical('------ CAM_SACKER Engine process started ------')

    # Write video
    if os.getenv('SAVE_VID') == 'TRUE':
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        vid_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), 'CAM_SACKER.mp4')
        videoWriter = cv2.VideoWriter(vid_path, fourcc, int(os.getenv('FPS_CAM_SACKER')), (1920, 1080))

    # Create color table & ROI
    np.random.seed(17)
    colours = np.random.randint(0, 256, size=(45, 3))
    engine_frames = deque(maxlen=20)
    roi_x1y1 = list(ast.literal_eval(os.getenv('ROI_CAM_SACKER')))[0]
    roi_x2y2 = list(ast.literal_eval(os.getenv('ROI_CAM_SACKER')))[1]
    SACKER_POS = list(ast.literal_eval(os.getenv('SACKER_POS')))
    PRODUCT_STABLE_FREQ = int(os.getenv('PRODUCT_STABLE_FREQ'))
    PRODUCT_RECOG_THRESHOLD = float(os.getenv('PRODUCT_RECOG_THRESHOLD'))
    count_no_product = 0

    # Create instance of Redis
    r = redis.Redis(os.getenv("REDIS_HOST"), int(os.getenv("REDIS_PORT")))

    # Create instance of VMS Manager
    vms = VMSManager()

    # Create instance of Event Manager
    event = EventManager()

    # Create instance of Product Detector
    detector = Detectron(os.getenv('PRODUCT_DETECTOR_GPU'), os.getenv('PRODUCT_CONFIG_PATH'),
                         os.getenv('PRODUCT_DETECTOR_MODEL_PATH'), float(os.getenv('PRODUCT_SCORE_THRESHOLD')))
    detector.setROI(roi_x1y1, roi_x2y2)

    # Create instance of Product Recognition
    recognition = Recognition(os.getenv('CAM_SHELF_GPU'), os.getenv('PRODUCT_RECOGNITION_MODEL_PATH'),
                              os.getenv('PRODUCT_GALLERY_PATH'), os.getenv('PRODUCT_GALLERY_ID_PATH'))

    # Get product information list
    product_infos = load_products_db_csv(os.getenv('PRODUCT_INFO_CSV'))
    is_check_out = False
    list_max_products = []
    freq_list_max_products = []
    time_list_max_products = []
    age_stable = 0

    # Initial check event
    num_model = len(os.getenv('LIST_CAM_ENGINE').split(','))
    num_loaded_model.value += 1
    frame_cnt = 0
    while True:
        if (num_loaded_model.value < num_model) or (cam_queue.qsize() <= 0):
            continue
        start_time = time.time()
        img_data = cam_queue.get()
        img_ori = img_data['frame']
        cur_time = int(img_data['timestamp'])

        # Retrieve trackers from fish-eye process
        trackers = global_tracks.from_redis(int(img_data['timestamp'] * 1000) + 200)
        if trackers is None: trackers = []
        local_ids = map_local_ids(trackers, SACKER_POS)
        #print('Time of CAM_SACKER: {}'.format(int(img_data['timestamp'] * 1000)))
        #local_ids = ['1_1584464951']

        # Get qr_signal
        qr_signal = int(r.get(name='qr_signal'))
        if (qr_signal > 0):
            bpo_logger.info('{} - Time received QR signal: {}'.format('CAM_SACKER', start_time))
            bpo_logger.info('{} - Time of VMS Frame: {}'.format('CAM_SACKER', img_data['timestamp']))

        # Get Detection results
        detector.setFrame(img_ori[roi_x1y1[1]:roi_x2y2[1], roi_x1y1[0]:roi_x2y2[0]], frame_cnt)
        dets = detector.getOutput()

        # Get Product recognition result
        results = recognition.getOutput(img_ori, dets)

        # Visualize product recognition at the sacker table
        for bbox in results:
            if bbox[-2] < PRODUCT_RECOG_THRESHOLD: color = colours[44].tolist()
            else: color = colours[int(bbox[-1]) % 45].tolist()
            tl = round(0.001 * (img_ori.shape[0] + img_ori.shape[1]) / 2)
            c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(img_ori, c1, c2, color, thickness=tl)
            tf = max(tl - 1, 1)
            label = str(product_infos[int(bbox[-1])]['name'])
            if product_infos[int(bbox[-1])]['label_id'] <= 17:
                product_id = str(product_infos[int(bbox[-1])]['label_id'] + 1)
            else:
                product_id = str(product_infos[int(bbox[-1])]['label_id'] + 3)
            if bbox[-2] < PRODUCT_RECOG_THRESHOLD:
                product_id = '-1'
                bbox[-1] = -1
                label = 'Non-product'
            id_label_percentage = '{}_{}_{:.2f}'.format(product_id, label, bbox[-2])
            t_size = cv2.getTextSize(id_label_percentage, 0, fontScale=tl / 4, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 2
            cv2.rectangle(img_ori, c1, c2, color, -1)
            cv2.putText(img_ori, id_label_percentage, (c1[0], c1[1] - 2), 0, tl / 4, [0, 0, 0], thickness=tf,
                        lineType=cv2.LINE_AA)
        cv2.putText(img_ori, 'Frame #{:d} ({:.2f}ms)'.format(frame_cnt, (time.time() - start_time) * 1000), (30, 30),
                    0, fontScale=0.7, color=(0, 255, 0), thickness=2)

        # Update list check-out products candidates
        if len(results) > 0:
            if (len(local_ids) > 0) and (not is_check_out):
                is_check_out = True
                start_time_check_out = cur_time
                local_ids_check_out = local_ids
                list_max_products = []
                freq_list_max_products = []
                time_list_max_products = []
                prev_dict_products = dict()
                age_stable = 0
                engine_frames.append(img_ori)
                time_product = img_data['timestamp']
            if is_check_out:
                count_no_product = 0
                product_ids = results[:, -1].astype(np.int)
                dict_products = dict()
                for product_id in product_ids:
                    if product_id < 0: continue
                    if product_id <= 17:
                        new_product_id = product_id + 1
                    else:
                        new_product_id = product_id + 3
                    dict_products[int(new_product_id)] = dict_products.get(int(new_product_id), 0) + 1
                if dict_products == prev_dict_products:
                    age_stable += 1
                else:
                    if age_stable > PRODUCT_STABLE_FREQ:
                        if prev_dict_products not in list_max_products:
                            list_max_products.append(prev_dict_products)
                            freq_list_max_products.append(age_stable)
                            time_list_max_products.append(time_product)
                        else:
                            freq_list_max_products[list_max_products.index(prev_dict_products)] += age_stable
                    prev_dict_products = dict_products
                    age_stable = 0
                    engine_frames.append(img_ori)
                    time_product = img_data['timestamp']
        else:
            if is_check_out: count_no_product += 1
            if age_stable > PRODUCT_STABLE_FREQ:
                if prev_dict_products not in list_max_products:
                    list_max_products.append(prev_dict_products)
                    freq_list_max_products.append(age_stable)
                    time_list_max_products.append(time_product)
                else:
                    freq_list_max_products[list_max_products.index(prev_dict_products)] += age_stable
                age_stable = 0
        # Finish checkout item
        if (is_check_out) and (count_no_product > PRODUCT_STABLE_FREQ):
            engine_frames.append(img_ori)
            max_product = 0
            most_age_stable = 0
            index_most_stable = -1
            for i in range(len(list_max_products)):
                if (time_list_max_products[i] - start_time_check_out) < (0.6 * (img_data['timestamp'] - start_time_check_out)):
                    total_products = sum(list_max_products[i].values())
                    if (max_product < total_products) or ((max_product == total_products) and (most_age_stable < freq_list_max_products[i])):
                        max_product = total_products
                        index_most_stable = i
                        most_age_stable = freq_list_max_products[i]

            if index_most_stable > -1:
                sacker_data = []
                for product_id, quantity in list_max_products[index_most_stable].items():
                    sacker_data.append({'item_id': product_id, 'quantity': quantity})
                save_events_toDB(vms, event, local_ids_check_out, sacker_data, int(time_list_max_products[index_most_stable]), engine_frames)
                count_no_product = 0
            is_check_out = False
            engine_frames.clear()

        engine_logger.info(
            "Frame Count: {} - CAM_SACKER Flow with FPS: {}".format(frame_cnt, 1. / (time.time() - start_time)))

        if os.getenv('SHOW_GUI_SACKER') == 'TRUE':
            cv2.imshow('Product Recognition at SACKER table', img_ori)
            cv2.waitKey(1)

        if os.getenv('SAVE_VID') == 'TRUE':
            videoWriter.write(img_ori)

        frame_cnt += 1

    cv2.destroyAllWindows()
    if os.getenv('SAVE_VID') == 'TRUE':
        videoWriter.release()

    engine_logger.critical('------ CAM_SACKER Engine process stopped ------')

def save_events_toDB(vms, event, local_ids, sacker_data, cur_time, engine_frames):

    cur_time_in_ms = cur_time * 1000
    # Save SACKER event to database
    if len(local_ids) > 0:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_name = '{}.mp4'.format(str(uuid.uuid1()))
        video_path = join(os.getenv('CROPPED_IMAGE_FOLDER'), video_name)
        writer_event = cv2.VideoWriter(video_path, fourcc, 5, (1920, 1080))
        for i in range(0, len(engine_frames)):
            writer_event.write(engine_frames[i])
        writer_event.release()

        # Cam sacker
        id_cam_cam_sacker_north = '_'.join(['ID', 'CAM_SACKER', 'NORTH'])
        id_cam_cam_sacker_south = '_'.join(['ID', 'CAM_SACKER', 'SOUTH'])
        videl_url_cam_sacker = DataTemplate.video_url(os.getenv('ID_CAM_SACKER'), 'CAM_SACKER')
        videl_url_cam_sacker_north = DataTemplate.video_url(os.getenv(id_cam_cam_sacker_north), 'CAM_SACKER_NORTH')
        videl_url_cam_sacker_south = DataTemplate.video_url(os.getenv(id_cam_cam_sacker_south), 'CAM_SACKER_SOUTH')
        video_url_engine = 'http://{}:{}/{}'.format(os.getenv('PUBLIC_IP'), os.getenv('CROPPED_IMAGE_PORT'), video_name)
        video_url_evidence = vms.get_video_url([videl_url_cam_sacker, videl_url_cam_sacker_north, videl_url_cam_sacker_south], cur_time_in_ms - 7000, cur_time_in_ms + 3000)
        data = DataTemplate.sacker_event(local_ids, cur_time, sacker_data, video_url_evidence, video_url_engine)
        #print(data)
        event.save_event(data)

def load_products_db_csv(db_path):
    df = pd.read_excel(db_path, index_col = False, header = None)
    results = []
    for index, row in df.iterrows():
        results.append(
            {'label_id': int(row[0]),
             'name': str(row[1])
             }
        )
    return results

def map_local_ids(trackers, SACKER_POS):
    localIDs_sacker = []
    local_id = None
    if len(trackers) == 0: return localIDs_sacker
    min_distance = 1000000000
    for trk in trackers:
        localID = trk[-2]
        if localID < 0:
            continue
        center = (trk[0] + trk[2]) / 2, (trk[1] + trk[3]) / 2
        vector = np.array([center[0] - SACKER_POS[0], center[1] - SACKER_POS[1]])
        distance = np.linalg.norm(vector)
        if min_distance > distance:
            min_distance = distance
            local_id = localID
            timestamp = trk[-3]

    if local_id is not None:
        local_id_time = concat_local_id_time(local_id, timestamp)
        localIDs_sacker.append(local_id_time)

    return localIDs_sacker
