#successfully training and testing model code of video recognition
import cv2
import numpy as np

# Function to find DNN model files (e.g., .prototxt and .caffemodel)
def find_dnn_file(filename, required=False):
    return filename  # Modify if you need to find files in a different way or check their existence.

# File paths for model weights and config files
weights = find_dnn_file("model/res10_300x300_ssd_iter_140000.caffemodel", required=True)
config = find_dnn_file("model/deploy.prototxt.txt", required=True)

if weights is None or config is None:
    raise Exception("Missing DNN test files (model/deploy.prototxt.txt and model/res10_300x300_ssd_iter_140000.caffemodel). Verify OPENCV_DNN_TEST_DATA_PATH configuration parameter.")

# Use OpenCV to instantiate the Caffe model
model = cv2.dnn_DetectionModel(weights, config)
model.setInputParams(size=(300, 300), mean=(104.0, 177.0, 123.0), scale=1.0)

# Path to your video file
video_path = "videoplayback.mp4"  # Replace with your video file path

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Loop over the frames of the video
while True:
    ret, frame = cap.read()  # Read the next frame from the video
    if not ret:
        print("End of video or error reading frame.")
        break

    # Perform face detection on the frame
    output = model.detect(frame, confThreshold=0.5)

    # Check the output format
    if len(output) == 2:
        faces, confidences = output
        class_ids = None  # No class IDs for a single-class detection
    elif len(output) == 3:
        faces, confidences, class_ids = output
    else:
        print("Unexpected output format")
        faces, confidences = [], []  # Default to empty in case of an error

    # Debug: Print faces and confidences to check their structure
    print("Faces structure:", faces)  # See what the faces array contains
    print("Confidences:", confidences)  # See the confidence values

    # Loop through the detected faces and draw bounding boxes
    if isinstance(faces, np.ndarray):
        for i in range(faces.shape[0]):  # Iterate over detected faces
            print("Detected face:", faces[i])  # Debugging: print each face
            
            # Directly access the values assuming they are in the format [x, y, w, h]
            if faces.ndim == 2 and faces.shape[1] == 4:
                x, y, w, h = faces[i]  # Each face should be [x, y, w, h]
                confidence = confidences[i]  # Confidence score for the face

                # Draw bounding box and confidence score on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence * 100:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("Unexpected face data:", faces[i])

    # Show the frame with detected faces
    cv2.imshow("Detected Faces", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

print("Video processing completed.")
