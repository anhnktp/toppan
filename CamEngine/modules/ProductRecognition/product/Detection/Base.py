"""
@This class implements the product detection
@Author: naviocean
"""

from abc import ABCMeta, abstractmethod

class DetectionBase(object, metaclass=ABCMeta):

    def __init__(self):

        """
        @The constructor
        @Parameters
        @   image_id: the id of frame image (ex: frame_001)
        @   image_src: the source of frame image, it maybe base64encoded or vector matrix encoded
        @   cam_id: the id of camera (ex: cam_01)
        """
        self.__cam_id = None
        self.__image_id = None
        self.__image_src = None


    def setCamId(self, cam_id):
        """
        @Set new camera id
        @parameters
        @   cam_id: the id of camera
        @return: void
        """
        self.__cam_id = cam_id

    def setImage(self, image_src, image_id):
        """
        @Set new image
        @parameters
        @   image_id: the id of frame image
        @   image_src: the source of frame image
        @return: void
        """
        self.__image_id = image_id
        self.__image_src = image_src

    @property
    def image(self):
        """
        @Get the frame image information
        @return: a object included:
            src: the source of frame image
            id: the id of current frame image
        """
        return {"src": self.__image_src, "_id": self.__image_id}

    @property
    def cam_id(self):
        """
        @Get the id of current camera
        @return:
            cam_id: the id of current camera
        """
        return self.__cam_id

    @abstractmethod
    def getOutput(self):
        """
        @Get output of detection model
        @return:
            cam_id: the id of camera
            image_id: the id of frame image
            image_src: the source of frame image
            boxes: array of objects
                box_id: the id of bounding box
                detection: object
                    box_geometry: [coordinates]
                    confidence: the result of detection model
                    label: the label of object
        """
        pass

    # @abstractmethod
    # def getOutput_cropped(self):
    #     """
    #     @Get output of detection model
    #     @return:
    #         cam_id: the id of camera
    #         image_id: the id of frame image
    #         image_src: the source of frame image
    #         boxes: array of objects
    #             box_id: the id of bounding box
    #             cropped_img: PIL image array
    #     """
    #     pass

