#training and testing model code for real-time face recognition
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

# Initialize the DNN face detection model
model = cv2.dnn.DetectionModel(weights, config)
model.setInputParams(size=(300, 300), mean=(104.0, 177.0, 123.0), scale=1.0)

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

if not cap.isOpened():
    print("[ERROR] Could not open video device.")
    exit()

print("[INFO] Starting video stream. Press 'q' to exit.")

while True:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break
    
    # Perform face detection
    output = model.detect(frame, confThreshold=0.3)  # Perform face detection

    # output is a single tuple containing faces and confidences
    faces = output[0]
    confidences = output[1]

    # Check if faces are detected
    if len(faces) > 0:
        for i in range(faces.shape[0]):  # Iterate over detected faces
            # Ensure that each detected face is unpacked correctly
            face = faces[i]
            if isinstance(face, np.ndarray) and len(face) == 4:
                x, y, w, h = face  # Each face should be [x, y, w, h]
                confidence = confidences[i]  # Confidence score for the face
                
                # Draw bounding box around detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence * 100:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("Unexpected face data:", face)
    else:
        print("[INFO] No faces detected.")

    # Display the output frame with bounding boxes
    cv2.imshow("Video - Face Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
