#Model Creation(Real time video recognition)
#This block loads the pre-trained model and processes each frame to detect faces using the DNN model.

import cv2
import imutils
import time
from imutils.video import VideoStream

def detect_faces_from_video():
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('model/deploy.prototxt.txt', 'model/res10_300x300_ssd_iter_140000.caffemodel')

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 1000 pixels
        frame = vs.read()

        # Check if the frame is valid
        if frame is not None:
            frame = imutils.resize(frame, width=1000)

            # Preprocess the frame for DNN model input
            blob, w, h = preprocess_frame(frame)

            # pass the blob through the network and obtain the detections and predictions
            net.setInput(blob)
            detections = net.forward()

            # Visualize the detections
            visualize_detection(frame, detections, w, h)

        # Wait for 'q' key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Cleanup
    cv2.destroyAllWindows()
    vs.stop()
#Summary:
#Data Preprocessing: Resizes each frame and converts it into a blob for input into the DNN model.
#Data Visualization: Draws bounding boxes around detected faces and displays the frame with bounding boxes.
#Model Creation: Loads the pre-trained model and processes video frames to detect faces.
#This structure helps modularize the different components of the code for easier maintenance or reusability.
