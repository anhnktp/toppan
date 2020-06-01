import time

import  numpy as np

from helpers.settings import *
from helpers.time_utils import concat_local_id_time
from .ActionBase import ActionBase
from .utils.shelves_touch_utils import count_in_shelf_area


class HandActionRecognition(ActionBase):

    def __init__(self, cam_type):
        super(HandActionRecognition, self).__init__(cam_type)
        self.last_time_have_event = None
        self.d_thresh = 200

    def detect_action(self, old_state, hands, trackers, item_boxes):
        # Select the state having max num of shelf_ids in the old_state
        shelf_ids_before = []
        for state in old_state:
            shelf_ids = count_in_shelf_area(state, item_boxes)
            if (len(shelf_ids) > len(shelf_ids_before)):
                shelf_ids_before = shelf_ids
        
        # if len(shelf_ids_before) > 0:     
        #     print(f'shelf_ids_before: {shelf_ids_before}')
        
        shelf_ids = count_in_shelf_area(hands, item_boxes)


        # if len(shelf_ids_before) > 0:
        #     if len(shelf_ids):
        #         for shelf_id, center in shelf_ids:
        #             distances = []
        #             for shelf_id_b4, center_b4 in shelf_ids_before:
        #                 if shelf_id==shelf_id_b4:
        #                     d = np.linalg.norm(np.array(center) - np.array(center_b4))
        #                     distances.append(d)
        #                 else:
        #                     new_shelves.append((shelf_id, center))
        #             if len(distances) > 0:
        #                 is_update = True
        #                 for distance in distances:
        #                     if distance < self.d_thresh:
        #                         is_update = False

        #                 if is_update:
        #                     new_shelves.append((shelf_id, center))       

        # else:
        #     new_shelves = shelf_ids

        if len(shelf_ids) > len(shelf_ids_before):
            new_shelves = list(set(shelf_ids).difference(set(shelf_ids_before)))

            return new_shelves

        return []
