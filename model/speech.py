from gtts import gTTS
import pyttsx3
import os
import tempfile
import time
import requests

def internet_available():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except requests.RequestException:
        return False

def speak_online(text, retries=3):
    for attempt in range(retries):
        try:
            tts = gTTS(text=text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                filename = tmp.name
            tts.save(filename)
            os.system(f'start /min wmplayer "{filename}"')
            time.sleep(len(text.split()) * 0.3)
            os.remove(filename)
            return True
        except Exception as e:
            print(f"[WARN] gTTS failed (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(1)
    return False

def speak_offline(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[ERROR] Offline TTS failed: {e}")

def speak(text):
    if internet_available():
        print("[INFO] Internet detected. Using Google TTS...")
        if not speak_online(text):
            print("[INFO] Falling back to offline speech.")
            speak_offline(text)
    else:
        print("[INFO] No internet. Using offline speech.")
        speak_offline(text)