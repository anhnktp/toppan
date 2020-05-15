"""
@This class implements data loader module
@Author: DQAN
"""
from abc import ABCMeta, abstractmethod

class ActionBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, cam_type):
        """
        @The constructor
        @parameter
        @   device:   cpu/gpu_id. Ex: 'cpu' or 'gpu:0'
        @   cam_type:   Ex 'CAM_SHELF_01' or 'CAM_SHELF_02' or 'CAM_SHELF_03'
        """
        self._cam_type = cam_type

    @abstractmethod
    def detect_action(self, old_state, hands, trackers, shelf_area):
        """
        @Detect action
        @parameter
        @   frame: frame_data
        @   trackers:   trackers from fish-eye engine
        """
        pass