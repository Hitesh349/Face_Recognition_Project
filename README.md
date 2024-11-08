# Face_Recognition_Project

/FaceRecognitionProject
│
├── README.md         # Documentation file providing an overview of the project                                                                                                                                                                                                                            .
|
├── Data                           # (Heading) Data
│   ├── /dataset                   # Directory for storing captured images for training and recognition
│   └── /output                    # (Heading) Output
│       ├── embeddings.pickle      # Stores the face embeddings
│       ├── label_encoder.pickle   # Contains the label encoder for names
│       ├── model.pickle           # The trained model for face recognition
│       └── recognizer.pickle      # The recognizer model used for predictions
|       └── le.pickle              # serialized object of a label encoder
|
├── Models                                               # (Heading) Models
│   ├── deploy.prototxt                                  # Configuration file for the face detection model
│   ├── res10_300x300_ssd_iter_140000.caffemodel         # Weights for the face detection model
│   └── openface_nn4.small2.v1.t7                        # Pre-trained embedding model

├── Data Pre-processing
│   ├── imagecapture.py -- (Image Capture and Data Preparation)       # Script to capture images from the webcam
|   ├── extract_embeddings.py -- (Extracting Face Embeddings)         # Script for extracting face embeddings from images
|   
├──  Model Creation
│   └── train_model.py -- (Training the Model)                                        # (Heading) Model Creation
|
├── Data Visualization
│   ├── recognize.py -- ( Real-time Face Image Recognition)             # Script to recognize faces in real-time Image
│   └── recognize_video.py (Real-time Face Video Recognition)           # Script to recognize faces in real-time video
|
|
├── Configuration and Documentation
|    ├── .gitignore                      # (heading) Configuration and Documentation
|    ├── .gitattributes                  # Defines attributes for paths in the Git repository
|    ├──  print.py                       # Script for printing results or logs



















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

 

 

 

 
