import cv2
import os
import threading
import time
import queue
import speech_recognition as sr
import google.generativeai as genai
from camera import open_camera, get_frame, release_camera
from face_recognise_module import load_known_faces, detect_face_boxes, recognize_face_embeddings
from speech import speak
from ai_vision import describe_image
import winsound

# ----------- CONFIG -------------
GEMINI_API_KEY = "AIzaSyCVWN3Vi7az_Gz28C3JROua2c4LfvWX5EY"  # Replace with your actual Gemini API key
DATA_DIR = "data"
GEMINI_MODEL_NAME = "gemini-1.5-flash"  # Adjust based on available Gemini model
# Removed DESCRIPTION_INTERVAL as automatic descriptions are disabled
# --------------------------------

# Initialize Google Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Shared variables
face_boxes = []
face_names = []
frame_queue = queue.Queue(maxsize=1)
description_queue = queue.Queue(maxsize=1)
lock = threading.Lock()
known_encodings, known_names = load_known_faces()
conversation_active = False
current_name = None
describe_flag = False  # New flag for on-demand description

def recognition_thread():
    global face_names
    while True:
        if frame_queue.empty():
            time.sleep(0.005)  # Reduced sleep for smoother face detection
            continue
        frame = frame_queue.get()
        with lock:
            current_boxes = list(face_boxes)
        face_crops = [frame[y:y+h, x:x+w] for (x, y, w, h) in current_boxes]
        names = recognize_face_embeddings(face_crops, known_encodings, known_names)
        with lock:
            face_names = names

def description_thread():
    while True:
        if description_queue.empty():
            time.sleep(1)
            continue
        frame = description_queue.get()
        description = describe_image(frame)
        speak(description)
        winsound.Beep(1000, 200)  # Beep after description

def speech_callback(recognizer, audio):
    global conversation_active, current_name, describe_flag
    print("[INFO] Listening...")
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"Heard: {text}")
        if "describe what you see" in text:
            describe_flag = True  # Trigger description
        if conversation_active:
            if "bye" in text or "quit" in text:
                speak("Goodbye!")
                winsound.Beep(1000, 200)  # Beep after goodbye
                conversation_active = False
                current_name = None
            else:
                reply = chat_with_gemini(text)
                speak(reply)
                winsound.Beep(1000, 200)  # Beep after reply
    except sr.UnknownValueError:
        pass  # Silently ignore unclear audio to reduce noise
    except Exception as e:
        print(f"Speech recognition error: {e}")

def chat_with_gemini(prompt):
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            [
                {"role": "user", "parts": ["You are a friendly AI assistant. Keep responses concise and helpful."]},
                {"role": "user", "parts": [prompt]}
            ],
            generation_config={"max_output_tokens": 150, "temperature": 0.7}
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, I'm having trouble connecting right now. Try again later."

def setup_environment():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created '{DATA_DIR}' folder")
    return True

if __name__ == "__main__":
    if not setup_environment():
        print("Setup failed.")
        exit(1)

    cap = open_camera()
    try:
        speak("Camera is ready. Starting live detection.")
        winsound.Beep(1000, 200)  # Beep after initial message

        threading.Thread(target=recognition_thread, daemon=True).start()
        threading.Thread(target=description_thread, daemon=True).start()

        r = sr.Recognizer()
        r.energy_threshold = 4000  # Higher threshold for better detection of speech start/end
        r.phrase_time_limit = 10  # Listen for up to 10 seconds per phrase
        m = sr.Microphone()
        with m as source:
            r.adjust_for_ambient_noise(source, duration=2)  # Longer adjustment for better calibration

        time.sleep(10)  # Wait 10 seconds before starting listening
        stop_listening = r.listen_in_background(m, speech_callback)

        frame_count = 0

        while True:
            try:
                frame = get_frame(cap)
            except Exception as e:
                print(f"Camera error: {e}")
                break

            frame_count += 1

            boxes = detect_face_boxes(frame)
            with lock:
                face_boxes = boxes
                if len(face_names) != len(boxes):
                    face_names = ["Processing..."] * len(boxes)

            if frame_count % 2 == 0:  # Increased frequency for smoother face detection
                if not frame_queue.full():
                    frame_queue.put(frame.copy())

            with lock:
                for i, (x, y, w, h) in enumerate(face_boxes):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    name = face_names[i] if i < len(face_names) else "Unknown"
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if name != "Unknown" and name != "Processing..." and not conversation_active:
                        speak(f"Hello {name}, what can I do for you?")
                        winsound.Beep(1000, 200)  # Beep after greeting
                        conversation_active = True
                        current_name = name

            # On-demand description check
            if describe_flag:
                if not description_queue.full():
                    description_queue.put(frame.copy())
                describe_flag = False

            try:
                cv2.imshow("Live Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception as e:
                print(f"OpenCV display error: {e}")
                break

    finally:
        stop_listening(wait_for_stop=False)
        release_camera(cap)