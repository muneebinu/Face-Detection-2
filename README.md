# Face Recognition System

## Project Overview

This project implements a real-time face recognition system using Python, OpenCV, and the `face_recognition` library. It detects and recognizes faces from a webcam video feed by comparing them to known faces stored in a dataset.

The system consists of two main scripts:

- **`trainmodel.py`**: Trains the model by processing face images stored in the dataset and generating encodings for each person. The result is saved as a model file.
- **`Face_recognization.py`**: Detects and recognizes faces in real-time using a webcam and the previously trained model.

---

## Features

- Face detection using Haar Cascade classifier.
- Face recognition using 128-dimensional face encodings.
- Real-time webcam input and video frame processing.
- Visual feedback with bounding boxes and recognition messages.
- Persistent model storage using `joblib`.

---

## Prerequisites

### Hardware
- Webcam (for real-time input).

### Software & Libraries
- Python 3.7 or later
- Python libraries:
  - `opencv-python`
  - `face_recognition`
  - `joblib`

---

## Files

- `haarcascade_frontalface_default.xml` – Face detection XML file (Haar Cascade).
- `static/faces/` – Directory for training images, organized by person.
- `static/face_recognition_model.pkl` – Trained face encodings model.
- `trainmodel.py` – Script to train the model.
- `Face_recognization.py` – Script to run real-time face recognition.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
 2.Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install opencv-python face_recognition joblib
Folder Structure
cpp
Copy
Edit
<project-root>/
├── static/
│   ├── faces/
│   │   ├── Person1/
│   │   ├── Person2/
│   ├── haarcascade_frontalface_default.xml
│   └── face_recognition_model.pkl
├── trainmodel.py
├── Face_recognization.py
└── README.md
Troubleshooting
Webcam Not Working?
Ensure your webcam is connected and working.

Try changing the camera index in cv2.VideoCapture(0).

Model Not Found?
Run trainmodel.py first.

Check the correct file path in Face_recognization.py.

Poor Recognition Accuracy?
Use higher-quality images during training.

Adjust the face distance threshold (default: 0.6) for stricter/looser matching.

Performance Issues?
Resize video frames in the code to lower resolution for faster processing:
