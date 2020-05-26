# Project9 - Toppan

 ExactReID is a project bult on Python3.7 for a Deeplearning Tracking technology
 It features: 
  - Person Detection and Tracking for 360-degree camera
  - Basket Detection for 360-degree camera
  - Pose Extraction and Action Recognition for Shelf camera
  - Product touch detection
  - Group people detection
  - Has attention to signage detection
  - Event Notification
  
## Quick start

First, you need to download config & model files from https://drive.google.com/drive/u/1/folders/1f-KMcQw10rC_5ldRRW3NDzJQnaGYxvpG

```
$ mkdir ./CamEngine/models
```
Put all downloaded files in directory  **./CamEngine/models/**
 
# Run flowing line to run AI Engine 

Before running, modify some parameter (rtsp stream, parameters..) on ./CamEngine/.env file
Run engine
```
# From ./CamEngine
$ cd ./CamEngine
$ python main.py
```