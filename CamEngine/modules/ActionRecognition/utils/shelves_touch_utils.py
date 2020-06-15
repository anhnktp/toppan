import cv2
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import numpy as np


def inPolygon(polygon, x):
    """Check if point `x` is inside `polygon`.
    # >>> inPolygon([[0, 0], [0, 3], [2, 2], [2, 1]], [1,1])
    True
    """
    point = Point(x)
    return polygon.contains(point)

def count_in_shelf_area(hands, item_boxes, vthresh = 100):
    shelves = []

    for hand in hands:
        # shelf_1: POLYGON ((612 251, 174 402, 102 86, 513 4, 612 251))
        for shelf_name, item_box in item_boxes.items():
            hand_center = hand[-1]
            hand_velo = hand[-2]
            #print(f'velo is {hand_velo}')
            #hand_vx2 = hand[-3]
            #hand_vy2 = hand[-2]
            curent_time = hand[-5]
            #print(f'time is {curent_time}')
            hand_id = hand[4]
            have_item_event = False

            if (not have_item_event) and (hand_center is not None) and inPolygon(item_box, hand_center) and hand_velo < vthresh:
                shelf_id = int(shelf_name.split('_')[-1])
                shelves.append((shelf_id, hand_id, hand_center, curent_time))
                have_item_event = True

            if have_item_event: # process the next hand
                continue

    return shelves


def min_ious(polygon, shelves_info):
    """
    min_iou = ratio between intersection area and smaller polygon area.
    Calc min_ious of a polygon (handbox) with shelves polygon
    """
    shelf_dict = shelves_info["shelf_dict"]
    ious = dict()

    # k:v = shelf_1: POLYGON ((651 474, 480 102, 129 291, 243 656, 651 474))
    for k, v in shelf_dict.items():
        # iou = polygon.intersection(v).area / polygon.union(v).area
        # Calculate min_iou
        iou = polygon.intersection(v).area / min(polygon.area, v.area)
        ious[k] = iou

    return ious


def box_to_polygon(box):
    """
    box = [x0, y0 , x1, y1]
    """
    x0 = box[0]
    y0 = box[1]
    x2 = box[2]
    y2 = box[3]

    x1 = x2
    y1 = y0
    x3 = x0
    y3 = y2

    points = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    polygon = Polygon(points)

    return polygon

def get_shelf_id(ious_dict, overlap_thresh=0.3):
    """
    Return the shelf id (loc) if it has the max IoU with a detected handbox
    """
    shelf_key = max(ious_dict.keys(), key=(lambda key: ious_dict[key]))
    
    if ious_dict[shelf_key] < overlap_thresh:
        return None, None

    shelf_id = shelf_key.split('_')[-1]
    shelf_id = int(shelf_id)
    
    return shelf_id, ious_dict[shelf_key]



def get_shelves_id(dets_per_frame, shelves_info, view):
    shelves_id = []
    boxes_center = []
    ious = []
    
    if len(dets_per_frame) > 0:
        for det in dets_per_frame:  # det = [x0, y0 , x1, y1, score]
            box = det[:4]
            polygon = box_to_polygon(box)
            ious_dict = min_ious(polygon, shelves_info)
            # print(ious_dict)
            shelf_id, iou = get_shelf_id(ious_dict)
            if shelf_id is not None:
                shelves_id.append(shelf_id)
                box_center = list(polygon.centroid.coords)
                box_center = box_center[0]
                boxes_center.append(box_center)
                ious.append(round(iou, 4))

    if len(shelves_id):   # if at least one hand is detected
        shelves_info_out = {"cam_id": view,
                            "shoppers_id": ["<BLANK>"] * len(shelves_id),
                            "process_id": 1201,
                            "shelves_id": shelves_id,
                            "unix_timestamp": -1,
                            "jst_timestamp": -1,
                            "boxes_center": boxes_center,
                            "ious": ious,}
    else:
        shelves_info_out = dict()

    return shelves_info_out