
DATA_DIR = 'C:/Users/win10/Downloads/aj/steps'

#import os

from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n-cls.pt")  # load a pretained model

# Use the model
results = model.train(data=DATA_DIR, epochs=45, imgsz=64)  # train the model
#scp -r /content/runs 'C:/Users/win10/Downloads/aj/steps'