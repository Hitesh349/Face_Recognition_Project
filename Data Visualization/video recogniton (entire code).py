#video recogniton (entire code)
import cv2
import numpy as np
import matplotlib.pyplot as plt  # For displaying images in Jupyter Notebook

def detectVidDNN(vidpath):
    # Load the pre-trained Caffe model and config
    modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "./model/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    
    # Capture video from the provided path
    video_capture = cv2.VideoCapture(vidpath)
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence < 0.5:
                continue

            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # Display the output frame in Jupyter Notebook using matplotlib
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
        plt.axis('off')  # Hide axes for better visualization
        plt.show()

        # Wait for 'q' key to exit (optional for Jupyter but not typically needed in Jupyter)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the video capture and close any OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()



