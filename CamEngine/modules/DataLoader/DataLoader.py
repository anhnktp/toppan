import time
import cv2
from multiprocessing import Process
from helpers.settings import *
from .LoaderBase import LoaderBase


class DataLoader(LoaderBase, Process):
    def __init__(self, rtsp_url, fps, cam_type, image_queue, num_loaded_model):
        super(DataLoader, self).__init__(rtsp_url, fps, cam_type)
        Process.__init__(self, daemon=True)
        self.queue_frame = image_queue
        self.num_loaded_model = num_loaded_model
        self.num_model = len(os.getenv('LIST_CAM_ENGINE').split(','))

    def connect(self):
        try:
            self._video = cv2.VideoCapture(self._rtsp_url)
            self._isOpen = True
            width = int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            engine_logger.info('Connected succesfully to {} !'.format(self._cam_type))
            engine_logger.info('{} - Width: {} - Height: {} - FPS: {}'.format(self._cam_type, width, height, self._fps))
        except Exception as ex:
            engine_logger.error("Failed connect to {} ! Error: {} !!!".format(self._cam_type, ex))
            raise ConnectionError("Failed connect to {} ! Error: {} !!!".format(self._cam_type, ex))

    def disconnect(self):
        self._video.release()
        self._isOpen = False
        engine_logger.info('Disconnected to {} !'.format(self._cam_type))

    def isOpened(self):
        return self._isOpen

    def run(self):
        engine_logger.critical('------ {} data loader process started ------'.format(self._cam_type))

        self.connect()
        self.disconnect()
        while self.isOpened():
            try:
                time.sleep(1 / self._fps)
                _, frame = self._video.read()
                if (self.num_loaded_model.value >= self.num_model) and (frame is not None):
                    timestamp = time.time()
                    self.queue_frame.put({'frame': frame, 'timestamp': timestamp})
            except Exception as ex:
                engine_logger.error("Something error ! : ".format(ex))
        self.disconnect()

        engine_logger.critical('------ {} data loader process stopped ------'.format(self._cam_type))
