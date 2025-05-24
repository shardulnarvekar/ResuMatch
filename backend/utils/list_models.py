import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load your API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def list_gemini_models():
    print("Available Gemini models:")
    for model in genai.list_models():
        print(model.name)

if __name__ == "__main__":
    list_gemini_models()

