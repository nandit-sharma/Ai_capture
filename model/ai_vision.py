import cv2
import base64
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyCVWN3Vi7az_Gz28C3JROua2c4LfvWX5EY"  # Replace with your actual Gemini API key
GEMINI_MODEL_NAME = "gemini-1.5-flash"  # Adjust based on available Gemini vision model

# Initialize Google Gemini client
genai.configure(api_key=GEMINI_API_KEY)

def describe_image(image):
    try:
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # Use Gemini API for image description
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            [
                {"role": "user", "parts": ["Describe the content of this image in a few sentences, focusing on people and objects present."]},
                {"role": "user", "parts": [{"mime_type": "image/jpeg", "data": jpg_as_text}]}
            ],
            generation_config={"max_output_tokens": 100, "temperature": 0.7}
        )
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini Vision failed: {e}")
        return "Unable to describe the scene due to API limitations. I can still detect faces locally."