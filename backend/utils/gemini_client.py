import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load your API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use correct model path
model = genai.GenerativeModel("models/gemini-1.5-flash")




def get_gemini_response(prompt: str) -> str:
    try:
        print("=== SENDING PROMPT TO GEMINI ===")
        response = model.generate_content(prompt)
        content = response.text.strip()
        print(f"✅ GEMINI RESPONSE RECEIVED — Length: {len(content)}")
        return content
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        raise Exception("Gemini request failed")

