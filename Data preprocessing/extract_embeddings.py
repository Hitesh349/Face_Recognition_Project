from imutils import paths
import numpy as np
import pickle
import cv2
import os

# File Paths
filedir = os.getcwd()
detector_path = os.path.join(filedir, "face_detection_model")
embedding_model_path = os.path.join(filedir, "openface_nn4.small2.v1.t7")
out_embeddings = os.path.join(filedir, "output", "embeddings.pickle")
dataset = os.path.join(filedir, "dataset")
confidence_limit = 0.5

# Load face detector
print("[INFO] loading face detector...")
protoPath = os.path.join(detector_path, "deploy.prototxt")
modelPath = os.path.join(detector_path, "res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load face embedding model
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model_path)

# Grab image paths
print("[INFO] embedding face data...")
imagePaths = list(paths.list_images(dataset))
knownEmbeddings = []
knownNames = []
total = 0

# Process each image
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_limit:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

# Serialize embeddings and names
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(out_embeddings, "wb") as f:
    f.write(pickle.dumps(data))

