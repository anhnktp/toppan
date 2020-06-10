"""
@This class implements the person Detection module
@Author: AnhVN
"""

from abc import ABCMeta, abstractmethod

class TrackerBase(object, metaclass=ABCMeta):

    def __init__(self):

        """
        @The constructor
        @Parameters
        @   frame_id: the id of frame image (ex: frame_001)
        @   frame_data: the source of frame image, it maybe base64encoded or vector matrix encoded
        @   cam_id: the id of camera (ex: cam_01)
        @   time_stamp: the timestamp correspond frame_id, frame_data from cam_id
        """
        self._timestamp = None

    def setTimestamp(self, timestamp):
        """
        @Set timestamp of self.__frame_id & self.__frame_data from cam_id
        @parameters
        @   timestamp: the timestamp correspond frame_id, frame_data from cam_id
        @return: void
        """
        self._timestamp = timestamp

    @property
    def time_stamp(self):
        """
        @Get the timestamp of self.__frame_id & self.__frame_data from cam_id
        @return:
            timestamp: the timestamp correspond frame_id, frame_data from cam_id
        """
        return self._timestamp

    @abstractmethod
    def update(self, dets, *args):
        """
        @Get output of Detection models
        @return:
            cam_id: the id of camera
            frame_id: the id of frame image
            frame_data: the source of frame image
            trackers: array of tracked objects
                tracked object
                    box_geometry: [coordinates]
                    trackID: the result of Detection models
        """
        pass