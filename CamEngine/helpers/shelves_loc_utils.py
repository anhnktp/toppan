
import os
import json

import cv2
import numpy as np

from shapely.geometry.polygon import Polygon

def get_shelves_loc(json_file):
    with open(json_file, 'r') as f:
        frame_info_dict = json.load(f)

    shelves_info_dict = frame_info_dict['frame_000.jpg']

    shelf_dict = dict()
    regions = shelves_info_dict['regions']
    for k, v in regions.items():
        label = v['region_attributes']['label']
        num_points = len(v['shape_attributes']['all_points_x'])
        points = []
        for i in range(num_points):
            x = v['shape_attributes']['all_points_x'][i]
            y = v['shape_attributes']['all_points_y'][i]
            point = (int(x), int(y))
            points.append(point)

        shelf_dict[label] = Polygon(points)

    shelves = [k for k, _ in shelf_dict.items()]
    shelf_to_idx = dict(zip(shelves, range(len(shelves))))
    idx_to_shelf = dict(zip(range(len(shelves)), shelves))

    shelves_info = {"shelf_dict": shelf_dict,
                    "shelf_to_idx": shelf_to_idx,
                    "idx_to_shelf": idx_to_shelf,}
    return shelves_info

def draw_shelves_polygon(img, shelves_info, color=(0, 255, 0)):
    shelves_dict = shelves_info["shelf_dict"]

    for k, v in shelves_dict.items():
        pts_lst = list(v.exterior.coords)
        polygon = np.array(pts_lst)
        polygon = np.array(polygon).reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(img, [polygon], True, (0,255,0), thickness=2)

        x = v.centroid.x
        y = v.centroid.y

        idx = k.split('_')[-1]

        cv2.putText(img, idx, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness=3)

if __name__ == '__main__':
    cwd = os.getcwd()
    shelf_loc_anno_dir = os.path.join(cwd, "shelves_loc_anno")
    json_fnames = ['shelves_loc_right.json', 'shelves_loc_left.json']
    
    for json_fname in json_fnames:
        json_file = os.path.join(shelf_loc_anno_dir, json_fname)
        shelves_info = get_shelves_loc(json_file)
        print(f"{json_fname}: ")
        print(shelves_info["shelf_to_idx"])
        print(shelves_info["idx_to_shelf"])
        shelf_dict = shelves_info["shelf_dict"]
        for k, v in shelf_dict.items():
            print(f'{k}: {v}')