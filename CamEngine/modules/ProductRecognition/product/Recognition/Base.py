"""
@This class implements the product recognition
@Author: naviocean
"""

from abc import ABCMeta, abstractmethod


class RecognitionBase(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        """
        @The constructor
        @Parameters
        @   image_src: the source of frame image
        @   box_id: the id of bounding box
        @   box_geometry: the coordinates of bounding box
        """
        self.__image_id = None
        self.__image_src = None
        self.__box_id = None
        self.__box_geometry = None

    def setImage(self, image_id, image_src):
        """
        @Set image
        @parameters:
        @   image_id: the id of new frame image
        @   image_src: the source of new frame image
        @return: void
        """
        self.__image_id = image_id
        self.__image_src = image_src

    def setBox(self, box_id, box_geometry):
        """
        @Set new box
        @parameters
        @   box_id: the id of new bounding box
        @   box_geometry: the coordinates of new bounding box
        @return: void
        """
        self.__box_id = box_id
        self.__box_geometry = box_geometry

    def getImage(self):
        """
        @Get current source of image
        @return: information of current image
        """
        return {"src": self.__image_src, "id": self.__image_id}

    def getBox(self):
        """
        @Get current box information
        @return:
            box_id: the id of current bounding box
            box_geometry: the coordinates of current bounding box
        """
        return {"box_id": self.__box_id, "box_geometry": self.__box_geometry}

    @abstractmethod
    def getOutput(self, img, bouding_boxes):
        """
        @Get result of recognition model
        @return: TOP5: array of objects
            label_id:
            confidence:
        """
        pass
