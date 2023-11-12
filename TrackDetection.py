import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import yaml
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

class Detector: #Parent for detection models
    def __call__(self, img) -> list: # list of objects [[x1,y1,x2,y2], ...]
        pass 
  
    def viz(self, img, objects):
        # Make a copy of the image
        img_with_detection = img.copy()
    
        # Draw rectangles of objects on the image
        for obj in objects:
            coordinates = obj.xyxy[0].tolist()
            coordinates = [round(x) for x in coordinates]
            top_left = tuple(coordinates[0:2])
            bottom_right = tuple(coordinates[2:])
            color = (0, 255, 0)  # Green color for rectangles
            thickness = 2
            img_with_detection = cv2.rectangle(img_with_detection, top_left, bottom_right, color, thickness)
        return img_with_detection

class NNDetector(Detector):
    def __init__(self, params):
        self.__model = YOLO(params['model']) 
        self.__objects = None
  
    def __call__(self, frame) -> list: # list of objects [[x1,y1,x2,y2], ...]
        self.__objects = self.__model.track(frame, persist=True)
        return self.__objects[0].boxes

yoloModel = NNDetector(data['YOLO'])

# Open the video file
video_path = "vtest.avi"
cap = cv2.VideoCapture(video_path)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (768,576))

# Loop through the video frames
while cap.isOpened():
     # Read a frame from the video
    success, frame = cap.read()
    if success:
        results = yoloModel(frame)
        detected_img = yoloModel.viz(frame, results)
        out.write(detected_img)
    else:
        print("can't open file")
        break
cap.release()
out.release()
