import cv2
import os
import joblib
import face_recognition
import serial
import time

# === Configuration ===
model_path = 'C:/Users/DELL/Desktop/Project/static/face_recognition_model.pkl'
arduino_port = 'COM3'  # Change based on your Arduino port
RECOGNITION_THRESHOLD = 0.55  # Strict threshold for accuracy

# === Arduino Connection ===
try:
    arduino = serial.Serial(arduino_port, 9600)
    time.sleep(2)
    print(f"[INFO] Connected to Arduino on {arduino_port}")
except Exception as e:
    print(f"[ERROR] Could not connect to Arduino: {e}")
    arduino = None

# === Extract Encoding ===
def extract_face_encoding(face_image):
    rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    return encodings[0] if encodings else None

# === Identify Face ===
def identify_face(face_image):
    try:
        data = joblib.load(model_path)
        known_encodings = data['faces']
        labels = data['labels']

        encoding = extract_face_encoding(face_image)
        if encoding is None:
            return None, 0.0

        distances = face_recognition.face_distance(known_encodings, encoding)
        min_index = distances.argmin()
        min_distance = distances[min_index]
        print(f"[DEBUG] Min distance: {min_distance:.3f}")

        if min_distance < RECOGNITION_THRESHOLD:
            confidence = max(0.0, 1.0 - min_distance) * 100
            return labels[min_index], confidence
        else:
            return None, 0.0
    except Exception as e:
        print(f"[ERROR] Recognition failed: {e}")
        return None, 0.0

# === Main Recognition Loop ===
def start_recognition():
    if not os.path.exists(model_path):
        print("[ERROR] Trained model not found.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam.")
        return

    print("[INFO] Starting face recognition... Press 'q' to quit.")
    last_status = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')

        if face_locations:
            for (top, right, bottom, left) in face_locations:
                face_image = frame[top:bottom, left:right]
                name, confidence = identify_face(face_image)

                if name:
                    text = f'{name} ({confidence:.1f}%)'
                    status = 'A'
                    if confidence >= 85:
                        color = (0, 255, 0)
                    elif confidence >= 70:
                        color = (0, 255, 255)
                    else:
                        color = (0, 100, 255)
                else:
                    text = 'UNKNOWN'
                    status = 'U'
                    color = (0, 0, 255)
                    confidence = 0.0

                # Send to Arduino
                if arduino and status != last_status:
                    arduino.write(status.encode())
                    last_status = status

                # Draw rectangle & label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # # Draw confidence bar
                # bar_x = left
                # bar_y = top - 30
                # bar_width = 150
                # bar_height = 10
                # filled = int((confidence / 100) * bar_width)
                # cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (180, 180, 180), 1)
                # cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), color, -1)
        else:
            # No face detected
            if arduino and last_status != 'N':
                arduino.write('N'.encode())
                last_status = 'N'

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

# === Start the System ===
start_recognition()
