from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def count_in_area(hands, item_boxes):

    shelves = []

    for hand in hands:
        for shelf_id, item_box in item_boxes:
            if len(item_box) == 0:
                continue
            left_hand, right_hand, bbox = hand
            have_item_event = False
            if (not have_item_event) and (left_hand is not None) and inPolygon(item_box, left_hand):
                shelves.append(shelf_id)
                have_item_event = True

            if (not have_item_event) and (right_hand is not None) and inPolygon(item_box, right_hand):
                shelves.append(shelf_id)
                have_item_event = True

            if have_item_event: # process the next hand
                continue

    return shelves

def inPolygon(polygon, x):
    """Check if point `x` is inside `polygon`.
    # >>> inPolygon([[0, 0], [0, 3], [2, 2], [2, 1]], [1,1])
    True
    """
    point = Point(x)
    polygon = Polygon(polygon)

    return polygon.contains(point)
