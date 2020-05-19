import time
from helpers.settings import *
from helpers.time_utils import concat_local_id_time
from .ActionBase import ActionBase
from .utils.check_in_area import count_in_area


class ActionRecognition(ActionBase):

    def __init__(self, cam_type):
        super(ActionRecognition, self).__init__(cam_type)
        self.last_time_have_event = None

    def detect_action(self, old_state, hands, trackers, item_boxes):
        # TODO: return local ID

        shelf_ids_before = []
        for state in old_state:
            shelf_ids = count_in_area(state, item_boxes)
            if (len(shelf_ids) > len(shelf_ids_before)):
                shelf_ids_before = shelf_ids

        shelf_ids = count_in_area(hands, item_boxes)

        # print(local_ids, max_hand_before)
        if len(shelf_ids) > len(shelf_ids_before):
            new_shelves = list(set(shelf_ids).difference(set(shelf_ids_before)))
            # new_shelves = (shelf_ids - shelf_ids_before)
            # # print("HAVE LOCAL ID", new_local_id, local_ids, max_hand_before)
            # if (self.last_time_have_event is None) or (time.time() - self.last_time_have_event > 1.0):
            #     self.last_time_have_event = time.time()
            #     local_ids = new_local_id
            #     # engine_logger.info("HAVE ITEM EVENT")
            #     return local_ids, in_shelf
            return new_shelves

        return []
