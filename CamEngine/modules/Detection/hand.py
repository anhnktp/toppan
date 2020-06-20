
import numpy as np


class HandCenter:
    last_id = -1
    def __init__(self, hand_center, confidence):
        # hands = [[x0, y0, x1, y1, round(score, 2), (xc, yc)], ]
        super().__init__()
        self.hand_center = hand_center
        self.confidence = confidence
        self.id = None
        
    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = HandCenter.last_id + 1
            HandCenter.last_id += 1

   
def get_distance(center1, center2):
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    return distance


def track_hands(previous_hands, current_hands, d_thresh=500):
    current_hands = sorted(current_hands, key=lambda hand: hand.confidence, reverse=True)
    mask = np.ones(len(previous_hands), dtype=np.int32)
    for current_hand in current_hands:
        best_matched_id = None
        best_matched_hand_id = None
        best_matched_distance = 1000
        for id, previous_hand in enumerate(previous_hands):
            if not mask[id]:
                continue
            distance = get_distance(current_hand.hand_center, previous_hand.hand_center)
            if distance < best_matched_distance:
                best_matched_distance = distance
                best_matched_hand_id = previous_hand.id
                best_matched_id = id
            if best_matched_distance < d_thresh:
                mask[best_matched_id] = 0
            else:
                best_matched_hand_id = None

            current_hand.update_id(best_matched_hand_id)
