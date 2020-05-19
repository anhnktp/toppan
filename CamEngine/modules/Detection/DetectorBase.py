"""
@This class implements the person Detection module
@Author: AnhVN
"""

from abc import ABCMeta, abstractmethod

class DetectorBase(object, metaclass=ABCMeta):

    def __init__(self):

        """
        @The constructor
        @Parameters
        @   frame_data: the source of frame image, it maybe base64encoded or vector matrix encoded
        @   cam_id: the id of camera (ex: cam_01)
        @   time_stamp: the timestamp correspond frame_id, frame_data from cam_id
        """
        self.__frame_data = None

    def setFrame(self, frame_data):
        """
        @Set new frame
        @parameters
        @   frame_id: the id of frame image (ex: frame_001)
        @   frame_data: the source of frame image, it maybe base64encoded or vector matrix encoded
        @return: void
        """
        self.__frame_data = frame_data

    def setROI(self, roi_x1y1, roi_x2y2):
        """
        @Set ROI of frame
        @parameters
        @   roi_x1y1: the coordinate of top-left point of ROI
        @   roi_x2y2: the coordinate of bottom-right point of ROI
        @return: void
        """
        self.__roi_x1y1 = roi_x1y1
        self.__roi_x2y2 = roi_x2y2


    @property
    def roi_x1y1(self):
        return self.__roi_x1y1

    @property
    def roi_x2y2(self):
        return self.__roi_x2y2

    @property
    def frame(self):
        """
        @Get the frame image information
        @return: a object included:
            frame_data: the source of frame image
            frame_id: the id of current frame image
        """
        return {"data": self.__frame_data}

    @abstractmethod
    def getOutput(self):
        """
        @Get output of Detection models
        @return:
            cam_id: the id of camera
            frame_id: the id of frame image
            frame_data: the source of frame image
            boxes: array of objects
                Detection: object
                    box_geometry: [coordinates]
                    confidence: the result of Detection models
        """
        pass