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
            print("[INFO] Speaking...")
            tts = gTTS(text=text, lang="en-us", slow=False)  # Updated for more natural, AI-like tone
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
        print("[INFO] Speaking...")
        engine = pyttsx3.init()
        # Set to female voice if available (AI-like, similar to ChatGPT/Gemini)
        voices = engine.getProperty('voices')
        female_voice = None
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():  # Common female voice like Microsoft Zira
                female_voice = voice.id
                break
        if female_voice:
            engine.setProperty('voice', female_voice)
        engine.setProperty('rate', 200)  # Slower, more natural pace
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