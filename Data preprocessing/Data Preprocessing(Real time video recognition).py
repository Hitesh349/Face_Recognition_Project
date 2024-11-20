#Data Preprocessing(Real time video recognition)
This block is responsible for preparing each frame by resizing it and converting it to a blob for input into the DNN model.

python
Copy code
import cv2
import numpy as np

def preprocess_frame(frame):
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))  # Normalize image
    return blob, w, h
