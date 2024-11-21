# Face_Recognition_Project

Configuration_and_Documentation
gitattributes.txt
gitignore.txt
print.py


Data_Preprocessing
Image_Recognition
Data_Preprocessing(Image recognition).py
Real_Time_Video_Recognition
Data_Preprocessing(Real time video recognition).py
Video_Recognition
Data_Preprocessing(Video recognition).py

Data_Visualization
Image_Recognition
Data_Visualization(Image recognition).py
Real_Time_Video_Recognition
Data_Visualization(Real time video recognition).py
Video_Recognition
Data_Visualization(Video recognition).py
Image_Recognition
image_recognition(entire code).py
image_recognitiondetection.py
Real_Time_Face_Recognition
real_time_face_recognition_code(entire_code).py
Video_Recognition
video_recognition(entire code).py
videorecognitiondetection.py


data
MOT16-11.mp4
group.jpg
groupangle.jpeg
groupangle.jpg
groupbig.jpg
groupsmall.jfif
groupsmall.jpg
tony.jpg
videoplayback.mp4

face_detection_model
deploy.prototxt
mmod_human_face_detector.dat
opencv_face_detector_uint8.pb
res10_300x300_ssd_iter_140000.caffemodel

file
adaboost.png
cnn.png
gradient.PNG
hog.gif
hog.png
hog1.png
hog2.png
hog3.png


model creation
Model Creation(Image recognition).py
Model Creation(Real time video recognition).py
Model Creation(Video recogniton).py
training and testing model code for real-time face recognition.py
training and testing model code of image recognition.py
training and testing model code of video recognition.py
README.md










Facial detection

 

Face recognition technology has rapidly evolved, utilizing machine learning and deep learning techniques for various applications, including security, user authentication, and personal identification. This project focuses on developing a robust face recognition system that allows for real-time identification using a webcam. It encompasses the entire pipeline, from image capture to model training and face recognition.

Facial detection has recently attracted increasing interest due to the multitude of applications that result from it. In this context, we have used methods based on machine learning that allows a machine to evolve through a learning process, and to perform tasks that are difficult or impossible to fill by more conventional algorithmic means.Face detection is performed from the images in the LFW data set.Before face detection in LFW images, multiple face detection performed in single images.Desired number of images are taken from the file in the LFW data set.

 

Related Work

Numerous face recognition projects exist, each contributing to the field through unique methodologies and applications. Some notable works include:

 

DeepFace: Introduced by Facebook, this deep learning model achieved significant accuracy on large datasets through the training of a neural network, emphasizing the importance of deep features for recognition.

 

OpenFace: Leveraging a deep neural network approach, OpenFace is capable of running in real-time and is open-source, offering accessibility for further research and development.

 

FaceNet: Google developed FaceNet to produce a compact representation of faces, allowing for clustering and identification through efficient embedding techniques. It set a benchmark for performance in face recognition tasks.

 

Despite these advancements, many existing projects face challenges in terms of model training data, adaptability to various lighting conditions, and the need for computational resources. The proposed project aims to address these limitations by providing a streamlined, user-friendly interface and a lightweight implementation that requires less computational power while maintaining high accuracy.

 

Methodology

The project is structured into several key stages, each contributing to the overall functionality of the face recognition system:

 
Summary of the Face Recognition Project
1. Introduction
This project implements face recognition and detection using deep learning-based models in Python with OpenCV. It demonstrates the application of face recognition in images, real-time video streams, and video files. It uses a pre-trained DNN model based on the SSD framework for face detection.

2. Tools and Libraries
Programming Language: Python
Libraries:
OpenCV: For image and video processing.
NumPy: For numerical operations.
Matplotlib: For visualization.
Imutils: For video stream resizing.
Model Files:
deploy.prototxt: Defines the structure of the DNN model.
res10_300x300_ssd_iter_140000.caffemodel: Pre-trained model weights.

3. Workflow

3.1 Image Face Detection (detectDNN)
Purpose: Detect faces in a static image.
Steps:
Load the DNN model from deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel.
Read the input image using OpenCV.
Preprocess the image to create a blob (resize and normalize).
Pass the blob to the DNN model for inference.
Iterate over the detections and:
Filter detections with confidence below 50%.
Draw bounding boxes around detected faces.
Annotate bounding boxes with confidence scores.
Display the image using Matplotlib.
Inputs: Path to the image (e.g., group.jpg, groupbig.jpg).
Output: Annotated image with detected faces.

3.2 Real-Time Video Face Detection
Purpose: Detect faces in real-time using a webcam.
Steps:
Load the DNN model.
Initialize the webcam video stream.
Continuously capture frames from the webcam.
Preprocess each frame and pass it to the model.
Detect faces and draw bounding boxes with confidence scores in real-time.
Display the processed frames in a window.
Terminate the process when the user presses the q key.
Inputs: Webcam stream.
Output: Real-time face detection displayed on the screen.

3.3 Video File Face Detection (detectVidDNN)
Purpose: Detect faces in a video file.
Steps:
Load the DNN model.
Capture video frames from the input video file.
For each frame:
Preprocess the frame to create a blob.
Pass the blob to the DNN model for inference.
Filter detections with confidence below 50%.
Draw bounding boxes and annotate them with confidence scores.
Display each processed frame using Matplotlib.
Release video resources after processing all frames.
Inputs: Path to the video file (e.g., videoplayback.mp4).
Output: Annotated frames from the video with detected faces.

4. Data Used
Images:
group.jpg, groupbig.jpg, groupangle.jpeg, groupsmall.jpg, tony.jpg.
Videos:
videoplayback.mp4.

5. Directory Structure
Configuration_and_Documentation: Project documentation files such as .gitattributes and .gitignore.
Data_Visualization: Scripts for visualizing face detection in various scenarios.
Image_Recognition: Scripts for detecting faces in images.
Real_Time_Face_Recognition: Real-time face recognition script.
Video_Recognition: Scripts for detecting faces in video files.
Data_Preprocessing: Preprocessing scripts for face recognition tasks.
Data: Dataset of images and videos used for testing.
Face_Detection_Model: Model files (deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel) for DNN inference.
Files: Supporting files

6.Data Preprocessing
Objective: Prepare data to be suitable for feeding into the model.
Steps:
Images and video frames were resized to dimensions of (300, 300) to meet model input requirements.
Mean subtraction (104.0, 177.0, 123.0) was applied to normalize input data.
Conversion of images from BGR (OpenCV format) to RGB was performed for visualization.
A blob was created using cv2.dnn.blobFromImage to preprocess the images for the DNN model.

7.Data Visualization
Static Images:
Detected faces were highlighted with bounding boxes on static images using cv2.rectangle.
Confidence scores for each detected face were displayed near the bounding boxes.
Processed frames were visualized using matplotlib for better integration with Jupyter Notebooks.
Video Frames:
Frames from video streams were processed in real time, with bounding boxes and confidence scores displayed dynamically using cv2.imshow.

8.Model Creation
Model Used:
A pre-trained DNN face detection model was used:
Weights: res10_300x300_ssd_iter_140000.caffemodel
Configuration: deploy.prototxt.txt
The model was based on the Single Shot MultiBox Detector (SSD) framework with ResNet-10 as the backbone.
Integration with OpenCV:
The model was loaded into OpenCV’s DNN module using cv2.dnn.readNetFromCaffe.
Input parameters like size, mean, and scale were set using model.setInputParams.

9. Testing
Static Image Testing:
The model was tested on individual images, detecting multiple faces and annotating them with bounding boxes and confidence scores.
Images of varying resolutions and angles were tested to ensure robustness.
Video Stream Testing:
Real-time face detection was implemented using webcam video streams (cv2.VideoCapture(0)) and pre-recorded videos.
The system dynamically processed each frame, detected faces, and displayed annotated results in real time.
10. Training Data
While the model was pre-trained, it utilized a large dataset during its original training. The model can be fine-tuned on custom datasets if desired.

The dataset for pre-training includes diverse faces across different ages, ethnicities, and conditions, ensuring generalization.

11.print.py
