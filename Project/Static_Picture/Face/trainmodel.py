import os
import cv2
import joblib
import face_recognition

# === Configuration ===
dataset_path = 'C:/Users/DELL/Desktop/Project/static/faces'  # Update this path
output_model_path = 'C:/Users/DELL/Desktop/Project/static/face_recognition_model.pkl'

known_encodings = []
labels = []

print("[INFO] Starting training...")

# === Load and encode faces ===
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            labels.append(person_name)

print(f"[INFO] Trained on {len(labels)} faces.")
joblib.dump({'faces': known_encodings, 'labels': labels}, output_model_path)
print(f"[INFO] Model saved to: {output_model_path}")
