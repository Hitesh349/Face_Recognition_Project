#Data Preprocessing(Video recognition)
#This block handles loading the video and preparing each frame by resizing and converting it to a blob.

python
Copy code
import cv2
import numpy as np

def preprocess_frame(frame):
    # Grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))  # Normalize image
    return blob, w, h
