"""
Created by sonnt at 2/6/20
"""
import ast
import os

import numpy as np

from helpers.action_recognition.check_in_area import inPolygon


class MappingManager():

    def __init__(self, cam_type, width, height):
        self.__shelf_areas = []
        self.__cam_type = cam_type
        self.__height = height
        self.__width = width
        for shelf in range(2):
            for index in range(2):
                cam_shelf_item = 'SHELF_{:02d}_{:02d}_{}'.format(shelf + 1, index + 1, self.__cam_type)
                area = list(ast.literal_eval(os.getenv(cam_shelf_item)))
                self.__shelf_areas.append(area)

    def get_shelf(self, point):
        for shelf_id, area in enumerate(self.__shelf_areas):
            if len(area) == 0:
                continue
            if inPolygon(area, point):
                return shelf_id
        return -1

    def mapping_local_id(self, hands, trackers):
        if self.__cam_type == 'CAM_360':
            return []
        hands_with_local_id = []
        for hand in hands:
            left_hand, right_hand, bbox = hand
            x1, y1, x2, y2 = bbox
            center_point = [(x1 + x2) / 2, (y1 + y2) / 2]
            shelf = self.get_shelf(center_point)

            # print("Bug: ", shelf, [tracker[-2:] for tracker in trackers])
            local_ids = []
            for tracker in trackers:
                timestamp, local_id, shelf_tracker = tracker[-3:]
                if (int(shelf) == int(shelf_tracker)) and (int(local_id) > 0):
                    local_ids.append((local_id, timestamp))
            hand = (left_hand, right_hand, bbox, local_ids, shelf)
            hands_with_local_id.append(hand)

        return hands_with_local_id

    def mapping_shelf(self, trackers):
        if self.__cam_type != 'CAM_360':
            return np.empty()
        trackers_with_shelf = []
        for tracker in trackers:
            x1, y1, x2, y2 = tracker[:4]
            point = ((x1 + x2) / 2, (y1 + y2)/2)
            shelf = self.get_shelf(point)
            tracker = np.append(tracker, shelf)
            trackers_with_shelf.append(tracker)
        return trackers_with_shelf
