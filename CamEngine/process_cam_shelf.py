import cv2
from helpers.settings import *
from helpers.video_utils import get_vid_properties, VideoWriter
from helpers.time_utils import *
from helpers.shelves_loc_utils import get_shelves_loc, draw_shelves_polygon
from modules.ActionRecognition import HandActionRecognition
from modules.Detection.Detector_yolov5 import HandDetector
from modules.Detection.Hand_velocity import HandVelocity
from modules.Visualization import HandVisualizer
from helpers.common_utils import CSV_Writer


def merge_results_n_save_to_csv(rets1, rets2, csv_path):
    def rearrange_results(results, cam_id):
        lst = []
        for ret in results:
            tpl = (cam_id, ret[0][0], int(ret[0][1]), ret[0][2], ret[0][3], ret[0][4])
            lst.append(tpl)
        return lst

    rets1_t = rearrange_results(rets1, 1)
    rets2_t = rearrange_results(rets2, 2)

    # Merge two results
    rets = rets1_t
    rets.extend(rets2_t)
    # Sort according time stamp
    rets.sort(key=lambda tup: tup[4])

    column_name = ['camera ID',
                   'shelf ID',
                   'hand_id',
                   'hand_coords',
                   'timestamp',
                   'timestamp(UTC - JST)']

    csv_writer = CSV_Writer(column_name, csv_path)
    for ret in rets:
        csv_writer.write(ret)
    csv_writer.to_csv()


def process_cam_shelf(camShelf_queue, num_loaded_model):
    shelf_touch_event = []
    # Get cam shelf id
    for i in range(1, 3):
        engine_logger.critical('------ CAM_SHELF_{:02} flow process started ------'.format(i))
        vid = cv2.VideoCapture(os.getenv('RTSP_CAM_SHELF_{:02}'.format(i)))
        width, height, fps, num_frames = get_vid_properties(vid)
        # Create video writer
        if os.getenv('SAVE_VID') == 'TRUE':
            videoWriter = VideoWriter(width, height, fps, os.getenv('OUTPUT_DIR'), 'CAM_SHELF_{:02}.mp4'.format(i))

        shelves_info = get_shelves_loc(os.getenv('CAM_SHELF_{:02}_POLYGONS_ANNO_FILE'.format(i)))

        item_boxes = shelves_info['shelf_dict']

        vis = HandVisualizer()

        # Hand detector and hand tracking
        detector = HandDetector(os.getenv('CAM_SHELF_GPU'),
                                os.getenv('HAND_CFG_PATH_YOLOV5'),
                                os.getenv('HAND_MODEL_PATH_YOLOV5'))


        roi_x1y1, roi_x2y2 = (0, 0), (width, height)
        detector.setROI(roi_x1y1, roi_x2y2)

        # Hand action recognition
        cam_type = 'CAM_SHELF_{:02}'.format(i)
        action = HandActionRecognition(cam_type)

        previous_hands_center = []
        old_state = []
        max_age = 13


        # Get start timestamp on video
        try:
            start_time = get_timestamp_from_filename(os.path.basename(os.getenv('RTSP_CAM_SHELF_{:02}'.format(i))), cam_type)
        except:
            start_time = time.time()
            engine_logger.info("Start time of video is set to current time !")

        # Hand velocity calculation
        handTracker = {}
        setHandId = set()
        handVelocity = HandVelocity(handTracker, setHandId)
        new_shelves_hand_touched_list = []
        mean_time = 0
        frame_cnt = 0
        while vid.isOpened():
            current_time = cv2.getTickCount()
            grabbed, img = vid.read()
            if not grabbed: break
            cur_time = start_time + vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            start_timestamp = time.time()
            # Get detection results
            frame = img[roi_x1y1[1]:roi_x2y2[1], roi_x1y1[0]:roi_x2y2[0]]
            org_frame = frame.copy()
            detector.setFrame(frame)
            hands = detector.getOutput(cur_time)

            hands = handVelocity.calculate_velocity(hands)
            # hand: [xmin, ymin, xmax, ymax, id, time, vx, vy, velo, (xc,yc)]

            trackers = []
            # Shelf touch detection
            states = old_state.copy()
            new_shelves_hand_touched = action.detect_action(states, hands, trackers, item_boxes)
            if len(old_state) == max_age:
                old_state.pop(0)

            old_state.append(hands)

            if len(new_shelves_hand_touched) > 0:
                h_time = convert_timestamp_to_human_time(cur_time)
                shelves = []
                for new_shelves_hand in new_shelves_hand_touched:
                    new_shelves_hand = list(new_shelves_hand)
                    new_shelves_hand.append(h_time)
                    shelves.append(new_shelves_hand)
                new_shelves_hand_touched_list.append(shelves)
            # Visualize handboxes
            vis.draw_boxes(hands, org_frame)

            # Draw shelves' polygon
            draw_shelves_polygon(org_frame, shelves_info)
            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            cv2.putText(org_frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                        (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
            engine_logger.info(
                'Frame Count: {} - CAM_SHELF_{:02} Flow with FPS: {}'.format(frame_cnt, i, 1. / (time.time() - start_timestamp)))

            videoWriter.write(org_frame)
            frame_cnt += 1

        shelf_touch_event.append(new_shelves_hand_touched_list)

        vid.release()
        videoWriter.release()
        engine_logger.critical('------ CAM_SHELF_{:02} Engine process stopped ------'.format(i))

    merge_results_n_save_to_csv(shelf_touch_event[0], shelf_touch_event[1], os.getenv('CSV_TOUCH_SHELF_PATH'))
    engine_logger.info('Created successfully CSV file of CAM_SHELF !')
