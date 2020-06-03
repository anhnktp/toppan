import os
import sys
import cv2
import numpy as np
from math import cos,sin
from keras import backend as K
from modules.Detection.DetectorBase import DetectorBase
from modules.Headpose.model import SSR_net_MT
import tensorflow as tf


class HeadposeDetector(DetectorBase):
    """
        Use FSA-Net Model
    """
    def __init__(self,ckpt_path):
        """ 
            Args:
                ckpt_path: weight of model
            Return:
                None
        """
        super(DetectorBase, self).__init__()
        # Fixed parameters 
        self.stage_num = [3,3,3]
        self.lambda_d = 1
        self.num_classes = 3
        self.image_size = 64
        self.weight = ckpt_path
        self.ad = 0.6
        self.model = self.load_model()

    def load_model(self):
        """
            Load model graph and weights ,set up config for Keras
            Return:
                model
        """
        K.clear_session()
        # make sure its testing mode
        K.set_learning_phase(0) 

        # Force the headpose model use CPU for computation
        config = tf.ConfigProto(device_count = {'CPU' : 1,
                                        'GPU' : 0})
        config.gpu_options.per_process_gpu_memory_fraction = 0.1

        session = tf.Session(config=config)
        K.set_session(session)

        model = SSR_net_MT(self.image_size, self.num_classes, self.stage_num, self.lambda_d)()
        model.load_weights(self.weight)

        return model

    def getOutput(self,input_img,box):
        '''
            Run the head pose estimation. Receive the bouding box of 1 face and 
            original image. Predict 3 angles roll, pitch, yaw.
            Args: 
                input_img: current frame 
                box: (np.arrary) bouding box of 1 face with order left, top, right, bottom 
                        (xmin,ymin,xmax,ymax)
            Return: 
                yaw,
                pitch
                roll
                person_face: personface cut from image based on bouding box
        '''

        (startX, startY, endX, endY) = box.astype("int")
        (img_h, img_w) = input_img.shape[:2]
        
        x1 = startX
        y1 = startY
        w = endX - startX
        h = endY - startY

        x2 = x1+w
        y2 = y1+h

        xw1 = max(int(x1 - self.ad * w), 0)
        yw1 = max(int(y1 - self.ad * h), 0)
        xw2 = min(int(x2 + self.ad * w), img_w - 1)
        yw2 = min(int(y2 + self.ad * h), img_h - 1)

        face = np.empty((self.image_size, self.image_size, 3))

        face[:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.image_size, self.image_size))
        face[:,:,:] = cv2.normalize(face[:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  

        face = np.expand_dims(face[:,:,:], axis=0)  

        p_result = self.model.predict(face)
        yaw, pitch, roll = p_result[0][0], p_result[0][1], p_result[0][2]

        return yaw,pitch,roll

    def draw_axis(self,input_img, bbox, yaw, pitch, roll, ad = 0.6, tdx=None, tdy=None, size = 80):
        '''
            Draw the axes for headpose estimation 
            Yaw, Pitch, Roll 
            Args: 
                input_img: np.array
                    frame
                bbox: list
                    contains 4 coordinates of face box
                yaw, pitch, roll: float
                    three angles of headpose
                ad, tdx, tdy, size: default setting coppied from author
                size: modify the size to increase/decrease the length of yaw,pitch,roll lines
            Returns:
                None
        '''
        (startX, startY, endX, endY) = bbox.astype("int")
        (img_h, img_w) = input_img.shape[:2]
        
        x1 = startX
        y1 = startY
        w = endX - startX
        h = endY - startY
        
        x2 = x1+w
        y2 = y1+h

        # no' expand cai' bounding dox ra
        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            tdx = (endX - startX)/2 + startX
            tdy = (endY - startY)/2 + startY

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy
        
        cv2.line(input_img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
        cv2.line(input_img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
        cv2.line(input_img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),3)
 
    def print_model(self):
        print (self.model.summary()) 

