"""
@This class implements data loader module
@Author: ANHVN
"""
from abc import ABCMeta, abstractmethod

class LoaderBase(object):
    __metaclass__ = ABCMeta
    def __init__(self, rtsp_url, fps, cam_type):
        """
        @The constructor
        @parameters
        @   rtsp_url: rtsp_url of camera stream (ex: rtsp://admin:12345@192.168.1.233:554/live)
        @   fps: frame per second (ex: 5)
        @   cam_type: type of camera (ex: CAM_360)
        """
        self._fps = fps
        self._rtsp_url = rtsp_url
        self._cam_type = cam_type
        self._isOpen = False  # status of camera stream connection

    @abstractmethod
    def connect(self):
        """
        @Connect to camera
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        @Disconnect to camera
        """
        pass

    @abstractmethod
    def isOpened(self):
        """
        @Return status of camera stream
        """
        pass
