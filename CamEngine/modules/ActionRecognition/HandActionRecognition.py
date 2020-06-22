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
        self.d_thresh = 140
        self.t_thresh = 2

    def detect_action(self, old_state, hands, trackers, item_boxes):
        shelf_ids_before = []
        for state in old_state:
            shelf_ids = count_in_shelf_area(state, item_boxes)
            if len(shelf_ids) > 0:
                shelf_ids_before.extend(shelf_ids)
        
        shelf_ids = count_in_shelf_area(hands, item_boxes)

        new_shelves = []
        if len(shelf_ids_before) > 0:
            if len(shelf_ids):
                for shelf_id, hand_id, center, t in shelf_ids:
                    distances = []
                    delt_ts = []
                    delt_ids = []
                    for shelf_id_b4, hand_id_b4, center_b4, t_b4 in shelf_ids_before:
                        if shelf_id==shelf_id_b4:
                            d = np.linalg.norm(np.array(center) - np.array(center_b4))
                            distances.append(d)
                            delt_ts.append(t-t_b4)
                            delt_ids.append(hand_id-hand_id_b4)
                        else:
                            if hand_id == hand_id_b4:
                                if (t - t_b4) > self.t_thresh:
                                    new_shelves.append((shelf_id, center, t))

                    if len(distances) > 0:
                        is_update = True
                        for distance, delt_t, delt_id in zip(distances, delt_ts, delt_ids):
                            if (distance < self.d_thresh or delt_t < self.t_thresh) and delt_id==0:
                                is_update = False

                        if is_update:
                            new_shelves.append((shelf_id, hand_id, center, t))

        else:
            new_shelves = shelf_ids

        return new_shelves
