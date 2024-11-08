# Face_Recognition_Project

/FaceRecognitionProject
│
├── README.md         # Documentation file providing an overview of the project
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
