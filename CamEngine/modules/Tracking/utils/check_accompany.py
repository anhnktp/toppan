import os

def compare_2bboxes_area(bbox1, bbox2, area_ratio_min=0.5, area_ratio_max=1.5):
    bbox1_width = max(bbox1[2] - bbox1[0], 0)
    bbox1_height = max(bbox1[3] - bbox1[1], 0)
    bbox1_area = bbox1_width * bbox1_height

    bbox2_width = max(bbox2[2] - bbox2[0], 0)
    bbox2_height = max(bbox2[3] - bbox2[1], 0)
    bbox2_area = bbox2_width * bbox2_height

    ratio = bbox1_area / bbox2_area

    if ratio >= area_ratio_min and ratio <= area_ratio_max:
        return True
    else:
        return False
