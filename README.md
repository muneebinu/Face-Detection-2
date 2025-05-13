Face Recognition System
Project Overview
This project implements a real-time face recognition system using Python, OpenCV, and the face_recognition library. It is designed to detect and recognize faces from a webcam video feed by comparing them to known faces stored in a dataset.

The system consists of two main scripts:

trainmodel.py: Trains the model by processing face images stored in the dataset and generating encodings for each person. The result is saved as a model file for later use.

Face_recognization.py: Uses a webcam to detect faces in real time and recognize them by comparing them to the trained model.

Features
Face detection using Haar Cascade classifier.

Face recognition using 128-dimensional face encodings.

Real-time webcam input and video frame processing.

Visual feedback via bounding boxes and recognition status in the video stream.

Persistent model storage using joblib.

Prerequisites
Hardware
Webcam (for capturing real-time video input).

Software & Libraries
Python 3.7 or later

Python packages:

opencv-python

face_recognition

joblib

Files
haarcascade_frontalface_default.xml: For face detection using Haar Cascade.

static/faces/: Directory containing training images, organized by person.

static/face_recognition_model.pkl: Saved model file generated after training.

trainmodel.py: Script for training the face recognition model.

Face_recognization.py: Script for real-time face detection and recognition.

Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
2. Create a Virtual Environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install opencv-python face_recognition joblib
4. Download Haar Cascade File
Download haarcascade_frontalface_default.xml from OpenCV GitHub and place it in the static/ directory.

Dataset Preparation
Organize your training images as follows:

python-repl
Copy
Edit
static/faces/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
├── Person2/
│   ├── image1.jpg
│   ├── image2.jpg
...
Each subfolder represents a person, and should contain multiple clear images of their face.

Usage
Step 1: Train the Model
Run the training script to encode the faces in the dataset and save the model:

bash
Copy
Edit
python trainmodel.py
This will process all images inside static/faces/ and save the trained encodings to static/face_recognition_model.pkl.

Make sure the path to the dataset is correctly set in trainmodel.py.

Step 2: Run Face Recognition
Start the real-time face recognition system:

bash
Copy
Edit
python Face_recognization.py
The webcam will open, and the script will:

Detect faces using Haar Cascade.

Recognize faces using the saved model.

Display:

A green rectangle and “WELCOME TO YOU” for recognized faces.

A red rectangle and “UNAUTHORIZED” for unknown faces.

Press q to quit the application.

Folder Structure
cpp
Copy
Edit
<project-root>/
├── static/
│   ├── faces/
│   │   ├── Person1/
│   │   ├── Person2/
│   │   └── ...
│   ├── haarcascade_frontalface_default.xml
│   └── face_recognition_model.pkl
├── trainmodel.py
├── Face_recognization.py
└── README.md
Troubleshooting
Webcam Not Working?
Make sure your webcam is properly connected.

Try changing the camera index in cv2.VideoCapture(0) if needed.

No Face Detected?
Check lighting and camera angle.

Ensure the Haar Cascade XML file path is correct.

Model Not Found?
Make sure you've run trainmodel.py first to generate face_recognition_model.pkl.

Recognition Not Accurate?
Use high-quality, well-lit images during training.

Adjust the face distance threshold (default is 0.6) in the recognition code.

Notes
Only one face is processed at a time by default. Modify the code to handle multiple faces if required.

For better performance or accuracy, consider reducing video resolution or using the CNN model (requires GPU).

Author
Muneeb


Acknowledgments
OpenCV for face detection and video processing.

face_recognition for facial feature encoding and comparison.

joblib for model serialization
