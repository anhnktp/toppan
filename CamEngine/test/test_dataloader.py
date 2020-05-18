import unittest
import os
from helpers.cam_data import get_engine_cams, get_evidence_cams
from modules.DataLoader import DataLoader
from multiprocessing import Queue
import time

class TestExtractFrame(unittest.TestCase):

    def test_dataloader(self):
        engine_cams = get_engine_cams()
        cam_data_loaders = []
        fnt = 0
        for cam_type in engine_cams:
            rtsp_url = engine_cams[cam_type]['RTSP_URL']
            fps = engine_cams[cam_type]['FPS']
            cam_loader = DataLoader(rtsp_url, fps, cam_type, Queue(int(os.getenv('QUEUE_SIZE'))))
            # cam_data_loaders.append(cam_loader)
            cam_data_loaders.append(cam_loader)
            # self.assertNotEqual(cam_loader.queue_frame.empty(), True)
        for cam_data_loader in cam_data_loaders:
            cam_data_loader.start()
            time.sleep(10)
            self.assertNotEqual(cam_data_loader.queue_frame.empty(), True)