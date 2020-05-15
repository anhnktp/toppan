import unittest
from CamEngine.Recognition.recognition import Recognition
import cv2
from CamEngine.Detection.yolo_detect import Detection
import time

#

class TestRecogintion(unittest.TestCase):
    # def testdetection(self):
    #     path = 'CamEngine/Detection/test/img96.jpg'
    #     img = cv2.imread(path)
    #
    #     A = Detection(image_id=1, image_src=img,cam_id=1)
    #     A.getOutput()
    #     print(A._result)
    #     self.assertGreater(len(A._result), 0)
    def testrecognition(self):
        path = 'CamEngine/Detection/test/img96.jpg'
        img = cv2.imread(path)
        detection = Detection()
        regcon = Recognition()
        detection.setImage(image_id=1, image_src=img)
        result = detection.getOutput()
        X = regcon.getOutput(img, result)
        print(X)

        self.assertGreater(len(X), 0)


if __name__ == '__main__':
    unittest.main()

# from CamEngine.Recognition.recognition import Recognition
# from CamEngine.Detection.yolo_detect import Detection
# import cv2
# import time
#
# if _name_ == '__main__':
#     regcon = Recognition()
#     detect = Detection()
#
#     links = ["/Users/phaihoang/Documents/datasets/satudora-product/all/product01_back_2_0001.jpg",
#              ]
#     for image in links:
#         start = time.time()
#         img = cv2.imread(image)
#         detect.setImage(image_id=1, image_src=img)
#
#         result = detect.getOutput()
#         print(result)
#
#         print(regcon.getOutput(image, result))
#         end = time.time()
#         print(end - start)