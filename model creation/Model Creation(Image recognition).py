#Model Creation(Image recognition)
import cv2
import numpy as np

def detectDNN(imgpath):
    modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"  # Path to Caffe model
    configFile = "./model/deploy.prototxt.txt"  # Path to config file
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)  # Load the model

    # Preprocess image
    blob, frame, h, w = preprocess_image(imgpath)

    # Set input for the network
    net.setInput(blob)

    # Perform forward pass to get detections
    detections = net.forward()  # Detections are returned here

    # Visualize detections
    visualize_detection(frame, detections, h, w)
