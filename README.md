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

 

Image Capture:

A Python script captures images through a webcam, allowing users to input their names. Images are saved in a designated dataset path, organized by user identity, reinforcing structured data collection.

Face Embedding Extraction:

Using OpenCV’s DNN module, facial embeddings are extracted from captured images. The script includes a face detector and a pre-trained embedding model to convert faces into numerical representations suitable for training.

Model Training:

Embeddings are labeled, and an SVM model is trained using scikit-learn, which learns to differentiate between various individuals based on their facial features. The model's performance is enhanced through stratified training/testing splits.

Real-Time Recognition:

A separate script implements real-time face recognition, utilizing the trained SVM model. It processes frames from the webcam, detects faces, extracts embeddings, and classifies them against the stored embeddings.

Efficiency and Scalability:

The entire system is designed for efficiency, using OpenCV for image processing and scikit-learn for model training. By focusing on modular design, users can easily update or replace components (e.g., embedding models or classifiers) as new advancements in technology emerge.

 

Here are some steps we did till now in our project-

1.Install Dependencies

Install necessary libraries: OpenCV, imutils, NumPy, scikit-learn, and TensorFlow.

2.Import Libraries

Import required libraries: cv2, os, random, numpy, pickle, imutils, and sklearn.

3.Set Up File Paths

Define file paths for models, datasets, and output directories to organize the project structure.

4.Load Face Detection Model

Load a pre-trained face detection model using OpenCV's DNN module (e.g., Caffe model).

5.Load Face Embedding Model

Load a pre-trained face embedding model (e.g., OpenFace) to convert detected faces into numerical vectors.

6.Collect and Label Images

Capture images from a video stream or dataset, allowing the user to input names for each person.

7.Data Preprocessing

Resize images to a uniform size (e.g., 300x300 pixels) for consistent input to the face detection model.
Normalize pixel values to prepare images for the embedding model.

8.Face Detection and Embedding Extraction

Process each image to detect faces and extract embeddings if confidence is above a specified threshold.

9.Serialize Data

Serialize the extracted embeddings and associated names using Python's pickle module for future use.

10.Data Visualization

Visualize sample images and their embeddings to ensure proper loading and preprocessing.
Optionally, plot histograms or scatter plots of embeddings to analyze distribution.

11.Train Recognition Model

Load the serialized embeddings, encode labels with LabelEncoder, and train a classifier (e.g., SVC) for face recognition.

12.Save Trained Model

Save the trained recognition model and label encoder to disk for later use.

13.Implement Real-time Face Recognition

Create a script to perform real-time face recognition using a webcam, displaying detected faces with bounding boxes and names.

14.Calculate Distance for Recognition

Implement a function to calculate the Euclidean distance between input face vectors and stored embeddings for identification.

15.Display Results

Show output frames with bounding boxes around detected faces and their recognized names in real-time.

16.Testing and Evaluation

Test the model on a separate test dataset to evaluate accuracy and performance metrics (e.g., precision, recall).
 

 

 

Team Details-

ANCEY -E23CSEU1784

HITESH-E23CSEU1782

PRIYANJAL-E23CSEU1797

 

 

 

 
