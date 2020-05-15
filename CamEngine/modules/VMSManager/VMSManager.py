from helpers.settings import *

class VMSManager():

    def __init__(self):
        self.vms_url_public = os.getenv('PUBLIC_IP') + ':' + os.getenv('VMS_PORT')
        self.vms_user = os.getenv('VMS_USER')
        self.vms_pwd = os.getenv('VMS_PWD')

    def get_video_url(self, cam_id, start_pos, end_pos):
        if isinstance(cam_id, str):
            return 'http://{}/media/{}.webm?pos={}&endPos={}'.format(
                self.vms_url_public, cam_id, start_pos, end_pos
            )
        elif isinstance(cam_id, list):
            for cam in cam_id:
                cam['video_url'] = 'http://{}/media/{}.webm?pos={}&endPos={}'.format(
                    self.vms_url_public, cam['cam_id'], start_pos, end_pos)
            return cam_id

    def get_frame_url(self, cam_id, timestamp):
        return 'http://{}/ec2/cameraThumbnail?cameraId={}&time={}'.format(
            self.vms_url_public, cam_id, timestamp)