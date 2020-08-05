from .settings import *


def set_cam(cam_id, rtsp, fps):
    return { "CAM_ID": cam_id, "RTSP_URL": rtsp, "FPS": int(fps) }

def get_engine_cams():
    engine_cams = dict()
    list_cam_engine = os.getenv('LIST_CAM_ENGINE').split(',')
    for cam_type in list_cam_engine:
        if 'CAM_360' in cam_type:
            fps_cam_type = '_'.join(['FPS', 'CAM_360'])
        elif 'CAM_SIGNAGE' in cam_type:
            fps_cam_type = '_'.join(['FPS', 'CAM_SIGNAGE'])
        else:
            fps_cam_type = '_'.join(['FPS', 'CAM_SHELF'])
        id_cam_type = '_'.join(['ID', cam_type])
        rtsp_cam_type = '_'.join(['RTSP', cam_type])
        engine_cams[cam_type] = set_cam(os.getenv(id_cam_type), os.getenv(rtsp_cam_type), os.getenv(fps_cam_type))

    return engine_cams


def get_evidence_cams():
    evidence_cams = dict()
    list_cam_evidence = os.getenv('LIST_CAM_EVIDENCE')
    for cam_type in list_cam_evidence:
        fps_cam_type = '_'.join(['FPS', 'CAM_IP'])
        id_cam_type = '_'.join(['ID', cam_type])
        rtsp_cam_type = '_'.join(['RTSP', cam_type])
        evidence_cams[cam_type] = set_cam(os.getenv(id_cam_type), os.getenv(rtsp_cam_type), os.getenv(fps_cam_type))

    return evidence_cams
