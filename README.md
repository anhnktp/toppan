# Prepare to run

Then, you need to download file ***models.zip*** from *https://drive.google.com/drive/u/0/folders/1f-KMcQw10rC_5ldRRW3NDzJQnaGYxvpG* 

(Put models in **./project9/CamEngine/models/**. If models  already exist, you can skip this step)
 
# Run flowing line to run all AI Engine 
Run engine
```
# From ./CamEngine
$ python main.py 
```

HYPER PARAMETERS in .env file:

```
# RTSP CONFIG
# Path to 3 Camera type video
RTSP_CAM_360 = path of 360 Camera
RTSP_CAM_SIGNAGE_1 = path of Signage Camera 1
RTSP_CAM_SIGNAGE_2 = path of Signage Camera 2
RTSP_CAM_SHELF_1 = path of Shelf Camera 1
RTSP_CAM_SHELF_2 = path of Shelf Camera 2

# CSV OUTPUT CONFIG
OUTPUT_DIR = folder for restoring CSV files

# HAND DETECTION CONFIG for YOLOV5
HAND_CFG_PATH_YOLOV5 = path of yaml YOLOv5 config file
HAND_MODEL_PATH_YOLOV5 = path of model YOLOv5
HAND_NMS_THRESHOLD_YOLOV5 = NMS thresh hold
HAND_SCORE_THRESHOLD_YOLOV5 = score threshold

# SELF TOUCH DETECTION PARAMETERS
VELOCITY_THRESHOLD = velocity threshold
DISTANCE_THRESHOLD = distance threshold
TIME_THRESHOLD = time threshold

# PATH TO ANNOTATION SHELF POLYGONS FILES
CAM_SHELF_01_POLYGONS_ANNO_FILE = path of shelf polygon define in CAM 01
CAM_SHELF_02_POLYGONS_ANNO_FILE = path of shelf polygon define in CAM 02
```