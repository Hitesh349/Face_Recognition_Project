#Model Creation(Video recogniton)
#This block initializes the DNN model and processes each frame from the video for face detection.


import cv2
import numpy as np

def detectVidDNN(vidpath):
    # Load the pre-trained Caffe model and config
    modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "./model/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)  # Load the model
    
    # Capture video from the provided path
    video_capture = cv2.VideoCapture(vidpath)
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Preprocess the frame
        blob, w, h = preprocess_frame(frame)

        # Pass the blob through the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()

        # Visualize detections
        visualize_detection(frame, detections, w, h)

        # Wait for 'q' key to exit (optional for Jupyter but not typically needed in Jupyter)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the video capture and close any OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()
#Summary:
#Data Preprocessing: Prepares the frames from the video for input into the DNN model by resizing and converting the frames into blobs.
#Data Visualization: Draws bounding boxes around detected faces and displays the frames with the bounding boxes.
#Model Creation: Loads the pre-trained DNN model and processes video frames to detect faces.
#Each part is separated into distinct functions and can be reused or modified independently.
