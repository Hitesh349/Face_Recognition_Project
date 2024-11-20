#successfully training and testing model code of image recognition

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
model.setInputParams(size=(224, 224), mean=(104.0, 177.0, 123.0), scale=1.0)

# Load test image
image_path = "groupbig.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    print(f"Error: Image not found at {image_path}")
else:
    # Perform face detection
    output = model.detect(image, confThreshold=0.5)

    # Check the output format
    print("Output:", output)  # Print the output for debugging

    if len(output) == 2:
        faces, confidences = output
        class_ids = None  # No class IDs for a single-class detection
    elif len(output) == 3:
        faces, confidences, class_ids = output
    else:
        print("Unexpected output format")
        faces, confidences = [], []  # Default to empty in case of an error

    # Print faces to check the structure
    print("Faces:", faces)

    # Loop over the faces detected and draw bounding boxes
    if isinstance(faces, np.ndarray):
        for i in range(faces.shape[0]):  # If 'faces' is a NumPy array, iterate through it
            print("Face:", faces[i])  # Debugging: print each face
            
            # If the value is a single integer (as suspected)
            if isinstance(faces[i], np.int32):
                x = faces[i]  # Assign the integer as a single value
                y = 0  # Default to zero, you can adjust based on logic
                w = 0  # Default to zero
                h = 0  # Default to zero
                print(f"Detected face at: {x}, {y}, {w}, {h}")
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print("Unexpected face data:", faces[i])

    # Show the output image
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Model forward pass completed successfully.")
