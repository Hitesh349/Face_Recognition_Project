#Data Preprocessing(Image recognition)

import cv2
import numpy as np

def preprocess_image(imgpath):
    # Load image from path
    frame = cv2.imread(imgpath)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Resize the image to match the input size expected by the model
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))  # Prepare the image as a blob for DNN

    return blob, frame, h, w
    
