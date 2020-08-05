import glob
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


def get_start_time(video_fname, cam_type):
    try:
        start_time = get_timestamp_from_filename(video_fname, cam_type)
        print(f"Get start time from {video_fname}")
    except:
        start_time = time.time()
        print("Start time is current time")

    return start_time


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


def process_cam_shelf(camShelf_queue, cam_type, num_loaded_model, global_tracks, n):
    engine_logger.critical('------ {} flow process started ------'.format(cam_type))

    video_base_dir = os.getenv('VIDEO_BASE_DIR')
    save_base_dir = os.getenv('SAVE_BASE_DIR')
    exp_name = 'separate'
    save_base_dir = os.path.join(save_base_dir, exp_name, 'test_cases')
    cases = ['case{}'.format(str(i + 1).zfill(2)) for i in range(25)]

    case = cases[n]

    case_dir = os.path.join(video_base_dir, case)
    case_save_dir = os.path.join(save_base_dir, case)
    if not os.path.exists(case_save_dir):
        os.makedirs(case_save_dir)

    videos_input = glob.glob(case_dir + '/*.mp4')
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

    # Hand detector and hand tracking
    detector = HandDetector(os.getenv('CAM_SHELF_GPU'),
                            os.getenv('HAND_CFG_PATH_YOLOV5'),
                            os.getenv('HAND_MODEL_PATH_YOLOV5'))

    roi_x1y1, roi_x2y2 = (0, 0), (width, height)
    detector.setROI(roi_x1y1, roi_x2y2)

    # Hand action recognition
    action = HandActionRecognition(cam_type)

    previous_hands_center = []
    old_state = []
    max_age = 13

    # Get start timestamp on video
    start_time = get_start_time(os.path.basename(video_input), cam_type)

    # Hand velocity calculation
    handTracker = {}
    setHandId = set()
    handVelocity = HandVelocity(handTracker, setHandId)

    new_shelves_hand_touched_list = []

    mean_time = 0
    while vid.isOpened():
        current_time = cv2.getTickCount()
        grabbed, img = vid.read()
        if not grabbed: break

        cur_time = start_time + vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

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
        vw.write(org_frame)
    print(new_shelves_hand_touched_list)
    return new_shelves_hand_touched_list
    vid.release()
    vw.release()


if __name__ == '__main__':
    csv_bdir = os.getenv('CSV_SAVE_BASE_DIR')
    camShelf_queue = []
    cam_type1 = 'CAM_SHELF_01'
    cam_type2 = 'CAM_SHELF_02'
    num_loaded_model = 1
    global_tracks = []

    for i in range(24, 25):
        rets1 = process_cam_shelf(camShelf_queue, cam_type1, num_loaded_model, global_tracks, i)
        rets2 = process_cam_shelf(camShelf_queue, cam_type2, num_loaded_model, global_tracks, i)

        csv_fname = 'case{}.csv'.format(str(i + 1).zfill(2))
        csv_path = os.path.join(csv_bdir, csv_fname)
        merge_results_n_save_to_csv(rets1, rets2, csv_path)