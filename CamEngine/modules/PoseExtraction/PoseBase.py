"""
@This class implements data loader module
@Author: DQAN
"""

from abc import ABCMeta, abstractmethod

class PoseBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, device):
        """
        @The constructor
        @parameters
        @   frame: frame data (base64 format)
        """
        self._gpu_number = device

    @abstractmethod
    def pre_process(self, frame):
        """
        @Pre-process image before interference
        """
        pass

    @abstractmethod
    def get_keypoints(self, frame):
        pass

    @abstractmethod
    def predict(self, frame):
        """
        @Extract keypoints
        @return: frame id, frame image
        """
        pass
