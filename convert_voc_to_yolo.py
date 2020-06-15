# coding: utf-8
import xml.etree.ElementTree as ET
import argparse
import os.path as osp
import os
from glob import glob
import cv2

def split_data(dataset_dir):
    basepath = osp.abspath(dataset_dir)

    ftrainval = open(osp.join(dataset_dir, 'ImageSets', 'Main', 'train.txt'), 'r')
    lines = ftrainval.readlines()
    train_txt = open(osp.join(dataset_dir, 'train.txt'), 'w')
    for line in lines:
        line = line.strip()
        jpg_path = osp.join(basepath, 'images', line + '.jpg')
        train_txt.write(jpg_path + os.linesep)
    train_txt.close()

    ftest = open(osp.join(dataset_dir, 'ImageSets', 'Main', 'test.txt'), 'r')
    lines = ftest.readlines()
    test_txt = open(osp.join(dataset_dir, 'test.txt'), 'w')
    for line in lines:
        line = line.strip()
        jpg_path = osp.join(basepath, 'images', line + '.jpg')
        test_txt.write(jpg_path + os.linesep)
    test_txt.close()

def gen_anno_txt(dataset_dir):
    for xml_path in glob(osp.join(dataset_dir, 'Annotations', '*.xml')):
        tree = ET.parse(xml_path)
        txt_path = xml_path.replace('Annotations', 'labels').replace('xml', 'txt')
        f = open(txt_path, 'w')
        jpg_path = xml_path.replace('Annotations', 'images').replace('xml', 'jpg')
        img = cv2.imread(jpg_path)
        width, height = img.shape[1], img.shape[0]
        for obj in tree.findall('object'):
            name = obj.find('name').text
            if name == 'non-person': continue
            x = float(obj.find('bndbox').find('xmin').text)
            y = float(obj.find('bndbox').find('ymin').text)
            w = float(obj.find('bndbox').find('xmax').text) - x
            h = float(obj.find('bndbox').find('ymax').text) - y
            x += w / 2
            y += h / 2
            line = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(names_dict[name], x / width, y / height, w / width, h / height)
            f.write(line)
        f.close()

def cls_to_dict(class_path):
    cls_dict = {}
    lines = open(class_path, 'r').readlines()
    cnt = 0
    for line in lines:
        line = line.strip()
        cls_dict[line] = cnt
        cnt += 1
    return cls_dict

if __name__ == '__main__':
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Convert VOC format to YOLOv3 format")
    parser.add_argument("-i",
                        "--dataset_dir",
                        default='/mnt/ssd2/Users/anhvn/ROI_PJ5_person360',
                        help="Path to the dataset VOC format",
                        type=str)
    parser.add_argument("-n",
                        "--class_path",
                        default='coco.names',
                        help="Path to the class name file",
                        type=str)
    args = parser.parse_args()

    names_dict = cls_to_dict(args.class_path)
    print(names_dict)
    assert osp.exists(args.dataset_dir), 'Invalid dataset path'

    os.makedirs(osp.join(args.dataset_dir, 'labels'), exist_ok=True)
    gen_anno_txt(args.dataset_dir)
    split_data(args.dataset_dir)
