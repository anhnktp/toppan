import math
import os

import cv2
import numpy as np
import torch
from modules.keypoints.keypoint import extract_keypoints as extract_keypoints1

from helpers.pose_extraction.keypoints import extract_keypoints, group_keypoints
from helpers.pose_extraction.load_state import load_state
from modules.Base.PoseBase import PoseBase
from modules.Model.pose import Pose
from modules.Model.with_mobilenet import PoseEstimationWithMobileNet
from helpers.action_recognition.check_in_area import inPolygon


class PoseExtraction(PoseBase):

    def __init__(self, device, model_path, roi_top, roi_bottom, roi_left, roi_right, item_boxes):
        super(PoseExtraction, self).__init__(device)
        self._stride = int(os.getenv("STRIDE_POSE"))
        self._pad_value = (0, 0, 0)
        self._img_mean = (128, 128, 128)
        self._img_scale = 1 / 256
        self._upsample_ratio = 4
        self._net_input_height_size = 256
        self._batch_size = 1
        self._num_kpts = 18
        self._net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(model_path, map_location='cpu')
        load_state(self._net, checkpoint)
        self._net = self._net.eval()
        self._net = self._net.cuda(self._gpu_number)

        self.l_wrist_idx = 7
        self.r_wrist_idx = 4

        self._roi_top = roi_top
        self._roi_bottom = roi_bottom

        self._roi_left = roi_left
        self._roi_right = roi_right

        self._item_boxes = item_boxes

    def pre_process(self, frame):
        # def normalize(img, img_mean, img_scale):
        img = np.array(frame, dtype=np.float32)
        img = (img - self._img_mean) * self._img_scale
        return img

    @staticmethod
    def pad_width(img, stride, pad_value, min_dims):
        h, w, _ = img.shape
        h = min(min_dims[0], h)
        min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
        min_dims[1] = max(min_dims[1], w)
        min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
        pad = []
        pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
        pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
        pad.append(int(min_dims[0] - h - pad[0]))
        pad.append(int(min_dims[1] - w - pad[1]))
        padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                        cv2.BORDER_CONSTANT, value=pad_value)
        return padded_img, pad

    def get_keypoints(self, frame):
        img = frame
        height, width, _ = img.shape

        scale = self._net_input_height_size / height
        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = self.pre_process(scaled_img)
        min_dims = [self._net_input_height_size, max(scaled_img.shape[1], self._net_input_height_size)]
        padded_img, pad = self.pad_width(scaled_img, self._stride, self._pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        tensor_img = tensor_img.cuda(self._gpu_number)

        stages_output = self._net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self._upsample_ratio, fy=self._upsample_ratio,
                              interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self._upsample_ratio, fy=self._upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def predict(self, frame):
        img = frame
        height, width, channels = img.shape
        num_keypoints = Pose.num_kpts
        heatmaps, pafs, scale, pad = self.get_keypoints(img)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self._stride / self._upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self._stride / self._upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            min_height_keypoints = height
            max_height_keypoints = 0
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    min_height_keypoints = min(min_height_keypoints, pose_keypoints[kpt_id, 1])
                    max_height_keypoints = max(max_height_keypoints, pose_keypoints[kpt_id, 1])

            # if (min_height_keypoints >= height * (self._roi_top - 0.1)) and (
            #         max_height_keypoints <= height * (self._roi_bottom + 0.1)):
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
            pose.draw(img)

        # cv2.line(img, (int(width * self._roi_left), 0), (int(width * self._roi_left), height), (0, 255, 0), 2)
        # cv2.line(img, (int(width * self._roi_right), 0), (int(width * self._roi_right), height), (0, 255, 0), 2)
        return current_poses, img, pose_entries, all_keypoints

    def predict_cython(self, img):
        height, width, channels = img.shape
        num_keypoints = Pose.num_kpts
        heatmaps, pafs, scale, pad = self.get_keypoints(img)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            # total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
            #                                          total_keypoints_num)
            total_keypoints_num += extract_keypoints1(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                      total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self._stride / self._upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self._stride / self._upsample_ratio - pad[0]) / scale

        return img, pose_entries, all_keypoints

    def check_person(self, pose):
        pair_check = [(2, 5), (3, 6), (8, 11)]
        for kpt_idx_1, kpt_idx_2 in pair_check:
            keypoint_1 = pose[kpt_idx_1]
            keypoint_2 = pose[kpt_idx_2]
            if (int(keypoint_1[0]) == -1) or  (int(keypoint_1[1]) == -1) or (int(keypoint_2[0]) == -1) or (int(keypoint_2[1]) == -1):
                continue
            distance = math.sqrt((keypoint_1[0] - keypoint_2[0]) ** 2 + (keypoint_1[1] - keypoint_2[1]) ** 2)
            if distance < 20:
                return False
        return True

    def get_hand_coord(self, frame, hand_score=0.5):
        """Method to get all pair of hands coordinates."""
        # current_poses, img, pose_entries, all_keypoints = self.predict(frame)
        img, pose_entries, all_keypoints = self.predict_cython(frame)
        height, width, channels = frame.shape

        hands = []
        for pose in pose_entries:
            if len(pose) == 0:
                continue

            is_in_shelf = True

            pose_keypoints = np.ones((self._num_kpts, 2), dtype=np.int32) * -1

            min_height_keypoints = 1e9
            max_height_keypoints = -1
            min_width_keypoints = 1e9
            max_width_keypoints = -1

            left_hand = None
            right_hand = None

            for idx in range(self._num_kpts):
                index_kpts = int(pose[idx])
                if index_kpts != -1:
                    keypoint = all_keypoints[index_kpts]
                    hand_x, hand_y, score = (int(keypoint[0]), int(keypoint[1]), keypoint[2])

                    pose_keypoints[idx, 0] = hand_x
                    pose_keypoints[idx, 1] = hand_y

                    point = (hand_x, hand_y)
                    _is_in_shelf = False
                    for item_box in self._item_boxes:
                        if len(item_box) == 0:
                            continue

                        if inPolygon(item_box, point):
                            _is_in_shelf = True
                            break

                    is_in_shelf = is_in_shelf & _is_in_shelf
                    min_width_keypoints = min(min_width_keypoints, hand_x)
                    max_width_keypoints = max(max_width_keypoints, hand_x)

                    min_height_keypoints = min(min_height_keypoints, hand_y)
                    max_height_keypoints = max(max_height_keypoints, hand_y)

                    if (idx == self.l_wrist_idx) and (score > hand_score):
                        left_hand = (hand_x, hand_y)

                    if (idx == self.r_wrist_idx) and (score > hand_score):
                        right_hand = (hand_x, hand_y)

            if (not is_in_shelf) \
                    and (min_width_keypoints >= width * self._roi_left) \
                    and (max_width_keypoints <= width * self._roi_right) \
                    and (min_height_keypoints >= height * self._roi_top) \
                    and (max_height_keypoints <= height * self._roi_bottom) \
                    and (max_height_keypoints - min_height_keypoints > 100) \
                    and (max_width_keypoints - min_width_keypoints > 100) \
                    and self.check_person(pose_keypoints):
                pose = Pose(pose_keypoints, pose[18])
                pose.draw(img)

                if (max_height_keypoints == -1) or (max_width_keypoints == -1):
                    bbox = [-1, -1, -1, -1]
                else:
                    bbox = [min_width_keypoints, min_height_keypoints, max_width_keypoints, max_height_keypoints]
                hands.append((left_hand, right_hand, bbox))

        return hands

    def draw_hand(self, frame, hands):
        for hand in hands:
            if hand[0] is not None:
                cv2.circle(frame, hand[0], 3, Pose.color, -1)

            if hand[1] is not None:
                cv2.circle(frame, hand[1], 3, Pose.color, -1)
