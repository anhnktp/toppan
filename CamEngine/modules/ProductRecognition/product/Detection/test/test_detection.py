import unittest
from ..yolo_detect import Detection
import cv2

class TestDetection(unittest.TestCase):
    def test_bounding_box(self):
        self.assertEqual(0, 0)
    def test_valid_frame(self):
        self.assertEqual(0,0)
    # No bounding box
    # Invalid frame

#
# class TestGetImgage(unittest.TestCase):
#     # Get Image id
#     # Get image
#

class TestDetection(unittest.TestCase):
    # def testdetection(self):
    #     path = 'CamEngine/Detection/test/img96.jpg'
    #     img = cv2.imread(path)
    #
    #     A = Detection(image_id=1, image_src=img,cam_id=1)
    #     A.getOutput()
    #     print(A._result)
    #     self.assertGreater(len(A._result), 0)
    def testdetection(self):
        path = 'CamEngine/Detection/test/img96.jpg'
        img = cv2.imread(path)
        detection = Detection()
        detection.setImage(image_id=1, image_src=img)
        detection.getOutput()

        print(detection._result['boxes'])
        self.assertGreater(len(detection._result), 0)


if __name__ == '__main__':
    unittest.main()