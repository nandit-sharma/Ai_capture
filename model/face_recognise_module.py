import os
import cv2
from deepface import DeepFace
import numpy as np

DATA_DIR = "data"

def load_known_faces():
    known_encodings = []
    known_names = []
    if not os.path.exists(DATA_DIR):
        return known_encodings, known_names
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(DATA_DIR, filename)
            try:
                embedding = DeepFace.represent(
                    img_path=path,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]
                known_encodings.append(embedding)
                known_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"[WARN] Could not load {filename}: {e}")
    return known_encodings, known_names

def enhance_brightness(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def detect_face_boxes(frame):
    if not hasattr(detect_face_boxes, "_cascade"):
        detect_face_boxes._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detect_face_boxes._cascade.empty():
        return []
    
    enhanced_frame = enhance_brightness(frame)
    gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
    
    boxes = detect_face_boxes._cascade.detectMultiScale(
        gray, 
        scaleFactor=1.03,  # Adjusted for smoother detection
        minNeighbors=6,    # Increased for more reliable detection
        minSize=(60, 60),  # Slightly smaller min size for better detection
        maxSize=(500, 500),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return list(boxes)  # Convert tuple to list

def recognize_face_embeddings(face_crops, known_encodings, known_names, threshold=0.4):
    results = []
    for face_img in face_crops:
        try:
            embedding = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]
            
            name = "Unknown"
            if known_encodings:
                distances = [cosine_distance(embedding, enc) for enc in known_encodings]
                min_dist = min(distances)
                min_idx = int(np.argmin(distances))
                if min_dist < threshold:
                    name = known_names[min_idx]
            
            results.append(name)
        except Exception:
            results.append("Unknown")
    return results

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1 - float(np.dot(a, b) / denom)