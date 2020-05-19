import os
import cv2
import argparse
import time
from multiprocessing import Process
from dotenv import load_dotenv

def set_cam(cam_id, rtsp, fps):
    return { "CAM_ID": cam_id, "RTSP_URL": rtsp, "FPS": fps }

def get_engine_cams():
    engine_cams = dict()
    list_cam_engine = os.getenv('LIST_CAM_ENGINE').split(',')
    if 'CAM_360' in list_cam_engine:
        engine_cams['CAM_360'] = set_cam(os.getenv('ID_CAM_360'), os.getenv('RTSP_CAM_360'),
                                         int(os.getenv('FPS_CAM_360')))
    if 'CAM_SHELF_01_MID' in list_cam_engine:
        engine_cams['CAM_SHELF_01_MID'] = set_cam(os.getenv('ID_CAM_SHELF_01_MID'), os.getenv('RTSP_CAM_SHELF_01_MID'),
                                              int(os.getenv('FPS_CAM_IP')))
    if 'CAM_SHELF_01_WEST' in list_cam_engine:
        engine_cams['CAM_SHELF_01_WEST'] = set_cam(os.getenv('ID_CAM_SHELF_01_WEST'), os.getenv('RTSP_CAM_SHELF_01_WEST'),
                                              int(os.getenv('FPS_CAM_IP')))
    if 'CAM_SHELF_01_EAST' in list_cam_engine:
        engine_cams['CAM_SHELF_01_EAST'] = set_cam(os.getenv('ID_CAM_SHELF_01_EAST'), os.getenv('RTSP_CAM_SHELF_01_EAST'),
                                              int(os.getenv('FPS_CAM_IP')))
    if 'CAM_SHELF_02_MID' in list_cam_engine:
        engine_cams['CAM_SHELF_02_MID'] = set_cam(os.getenv('ID_CAM_SHELF_02_MID'), os.getenv('RTSP_CAM_SHELF_02_MID'),
                                              int(os.getenv('FPS_CAM_IP')))
    if 'CAM_SHELF_02_WEST' in list_cam_engine:
        engine_cams['CAM_SHELF_02_WEST'] = set_cam(os.getenv('ID_CAM_SHELF_02_WEST'), os.getenv('RTSP_CAM_SHELF_02_WEST'),
                                              int(os.getenv('FPS_CAM_IP')))
    if 'CAM_SHELF_02_EAST' in list_cam_engine:
        engine_cams['CAM_SHELF_02_EAST'] = set_cam(os.getenv('ID_CAM_SHELF_02_EAST'), os.getenv('RTSP_CAM_SHELF_02_EAST'),
                                              int(os.getenv('FPS_CAM_IP')))
    if 'CAM_SHELF_03' in list_cam_engine:
        engine_cams['CAM_SHELF_03'] = set_cam(os.getenv('ID_CAM_SHELF_03'), os.getenv('RTSP_CAM_SHELF_03'),
                                              int(os.getenv('FPS_CAM_IP')))
    if 'CAM_SACKER_EAST' in list_cam_engine:
        engine_cams['CAM_SACKER_EAST'] = set_cam(os.getenv('ID_CAM_SACKER_EAST'), os.getenv('RTSP_CAM_SACKER_EAST'),
                                              int(os.getenv('FPS_CAM_IP')))
    return engine_cams


def get_evidence_cams():
    evidence_cams = dict()
    # CAM_ENTRANCE
    evidence_cams['CAM_ENTRANCE'] = set_cam(os.getenv('ID_CAM_ENTRANCE'), os.getenv('RTSP_CAM_ENTRANCE'), int(os.getenv('FPS_CAM_IP')))
    # CAM_EXIT
    evidence_cams['CAM_EXIT'] = set_cam(os.getenv('ID_CAM_EXIT'), os.getenv('RTSP_CAM_EXIT'), int(os.getenv('FPS_CAM_IP')))
    # CAM_WEST
    evidence_cams['CAM_WEST'] = set_cam(os.getenv('ID_CAM_WEST'), os.getenv('RTSP_CAM_WEST'), int(os.getenv('FPS_CAM_IP')))
    # CAM_EAST
    evidence_cams['CAM_EAST'] = set_cam(os.getenv('ID_CAM_EAST'), os.getenv('RTSP_CAM_EAST'), int(os.getenv('FPS_CAM_IP')))
    return evidence_cams

def save_video(rtsp_url, fps, cam_type, scenario_name):
    vid = cv2.VideoCapture(rtsp_url)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_ori = int(vid.get(cv2.CAP_PROP_FPS))
    output_path = scenario_name + '/' + cam_type + '.mp4'
    videoWriter = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print('Recording video from {} to ./{}. FPS original is {}'.format(cam_type, output_path, fps_ori))
    while vid.isOpened():
        _, img = vid.read()
        time.sleep(1 / fps)
        videoWriter.write(img)
    videoWriter.release()
    vid.release()

if __name__ == '__main__':
    # Config dotenv
    load_dotenv('engine.env')
    parser = argparse.ArgumentParser(description="Script for recording scenario of AWL Demo room in format <cam_type>.mp4")
    parser.add_argument("-d","--save_dir", type=str, default="scenario01", help="The path to save <cam_type>.mp4 to dir")
    args = parser.parse_args()
    scenario_dir = args.save_dir

    if not os.path.isdir(scenario_dir):
        os.mkdir(scenario_dir)

    engine_cams = get_engine_cams()
    evidence_cams = get_evidence_cams()
    list_processes = []

    for cam_type in engine_cams:
        rtsp_url = engine_cams[cam_type]['RTSP_URL']
        fps = engine_cams[cam_type]['FPS']
        p = Process(target=save_video, args=(rtsp_url, fps, cam_type, scenario_dir))
        list_processes.append(p)

    for cam_type in evidence_cams:
        rtsp_url = evidence_cams[cam_type]['RTSP_URL']
        fps = evidence_cams[cam_type]['FPS']
        p = Process(target=save_video, args=(rtsp_url, fps, cam_type, scenario_dir))
        list_processes.append(p)

    print('Start recording scenario! Press Ctrl + C to stop recording ...')

    for process in list_processes:
        process.start()

    for process in list_processes:
        process.join()
