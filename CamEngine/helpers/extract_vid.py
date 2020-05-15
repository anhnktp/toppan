import argparse
import os
from os.path import join
import glob
import cv2
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for extract frames from video !')
    parser.add_argument("-i", "--input_dir", type=str, default="scenario01", help="The path to directory contains scenarios")
    args = parser.parse_args()
    scenario_dir = args.input_dir
    evidence_cam = ['CAM_EAST.mp4', 'CAM_WEST.mp4']
    for sce in glob.glob(join(scenario_dir,'*')):
        if 'sce0' not in sce: continue
        for vid_path in glob.glob(join(sce, '*')):
            name = os.path.basename(vid_path)
            if name not in evidence_cam: continue
            name = name.lower()[0:-4]
            frame_dir = join(sce, name)
            if os.path.exists(frame_dir):
                shutil.rmtree(frame_dir)
            os.mkdir(frame_dir)
            vid = cv2.VideoCapture(vid_path)
            cnt = 0
            while vid.isOpened():
                _, img = vid.read()
                if img is None: break
                if cnt % 10 == 0:
                    img_name = name + '_{:04n}.jpg'.format(cnt)
                    img_path = join(frame_dir, img_name)
                    print(img_path)
                    cv2.imwrite(img_path, img)
                cnt += 1
            vid.release()