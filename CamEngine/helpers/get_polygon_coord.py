"""Script to draw polygons and return corressponding coordinates"""
import cv2
import numpy as np
import argparse
from collections import defaultdict


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the coordinate
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])
        cv2.circle(image, (x, y), 4, (0, 255, 255), thickness=-1, lineType=8, shift=0)


def save_coord():
    global save_data
    global refPt
    global image
    global last_saved_image
    area = ''

    pts = np.reshape(refPt, (-1, 1, 2))
    cv2.polylines(image, [pts], True, (0, 255, 255), 2)
    cv2.imshow("image", image)
    cv2.waitKey(1)

    while(area == ''):
        print('Enter area name: ', end='')
        area = input()

    if area != '':
        save_data[area] = refPt
        print(f'Saved coordinate for area{area}!')
        last_saved_image = image.copy()

    refPt = []


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=False, help="Path to the image/video")
    args = ap.parse_args()

    refPt = []
    save_data = defaultdict(int)
    image = []

    # load the image, clone it, and setup the mouse callback function
    # RTSP_CAM_360='rtsp://admin:2de087aa16bb2768578595208f2f7eda@39.110.250.100:7001/cbd3ca1e-df48-1ba9-ccc3-f6657a27cab3'
    RTSP_CAM_360='/Users/anhvu/PycharmProjects/YOLOv3_TensorFlow/CAM_360.mp4'
    RTSP_CAM_SHELF_01 = '/Users/anhvu/PycharmProjects/YOLOv3_TensorFlow/02_area1_shelf_left_2020_04_15_17_18_06.mp4'
    RTSP_CAM_SHELF_02 = '/Users/anhvu/PycharmProjects/YOLOv3_TensorFlow/01_area1_shelf_right_2020_04_15_17_18_06.mp4'
    RTSP_CAM_SHELF_03 = 'rtsp://admin:admin@192.168.8.124:554/stream2'
    RTSP_CAM_ENTRANCE = 'rtsp://admin:admin@192.168.8.121:554/stream2'
    RTSP_CAM_EXIT = 'rtsp://admin:admin@192.168.8.120:554/stream2'
    RTSP_CAM_LEFT = 'rtsp://admin:admin@192.168.8.126:554/stream2'
    RTSP_CAM_RIGHT = 'rtsp://admin:admin@192.168.8.125:554/steam2'
    RTSP_CAM_WEST = 'rtsp://admin:2de087aa16bb2768578595208f2f7eda@192.168.8.128:7001/128c9a24-e1ab-ad72-d6c0-798f2ab7e4da?pos=2020-02-13T16:34:00'
    RTSP_CAM_EAST = 'rtsp://admin:2de087aa16bb2768578595208f2f7eda@192.168.8.128:7001/0106d45f-4adc-d780-ff92-93a31c9d6c73?pos=2020-02-13T16:34:00'

    try:
        cap = cv2.VideoCapture(RTSP_CAM_360)
        _, image = cap.read()
    except BaseException:
        image = cv2.imread(args.input)

    clone = image.copy()
    last_saved_image = image.copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a key/mouse actions
        # _, image = cap.read()
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the key 's' is pressed, save shape coordinates
        if key == ord("s"):
            save_coord()

        # if the key 'c' is pressed, clear current points
        if key == ord("c"):
            refPt = []
            image = last_saved_image.copy()

        # if the 'r' key is pressed, reset all drawn shape
        if key == ord("r"):
            image = clone.copy()
            last_saved_image = clone.copy()
            save_data = defaultdict(int)
            print('Reset all saved data!')

        # if the 'q' key is pressed, quit
        if key == ord("q"):
            break
        print(save_data)
    if len(save_data) != 0:
        print(save_data)
        print('All saved coordinates:')
        [print(f'area{one}: {save_data[one]}') for one in save_data]
    else:
        print('No saved data!')

    # close all open windows
    cv2.destroyAllWindows()
