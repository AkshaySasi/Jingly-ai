import os
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from utils.musicgen_run import generate_jingle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ Missing GEMINI_API_KEY in .env file")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

class JingleRequest(BaseModel):
    brand_name: str
    motto: str
    product_brief: str
    genre: str
    reference_audio_path: Optional[str] = None

def build_ai_prompt(data: JingleRequest) -> str:
    parts = [
        f"Brand Name: {data.brand_name}",
        f"Motto: {data.motto}",
        f"Product Description: {data.product_brief}",
        f"Preferred Genre: {data.genre}",
        "Instruction: Generate an instrumental music prompt (no lyrics) that reflects a healthy and happy brand vibe, suitable for a 30-second jingle.",
        "Important: The result must be a vivid, short, MusicGen-compatible instrumental music prompt without lyrics.",
        "Avoid text suggesting lyrics or singing. Focus on musical mood, instruments, tempo, and style."
    ]
    return "\n".join(parts)

@app.post("/generate-jingle")
def generate_jingle_api(data: JingleRequest):
    prompt = build_ai_prompt(data)
    print("🔹 Prompt sent to Gemini:\n", prompt)
    response = model.generate_content(prompt)
    musicgen_prompt = response.text.strip()
    print("\n✅ Final MusicGen Prompt:\n", musicgen_prompt)

    output_path = generate_jingle(musicgen_prompt, data.reference_audio_path)
    return {
        "message": "🎵 Jingle generated successfully!",
        "output_path": output_path,
        "musicgen_prompt": musicgen_prompt
    }


# Uvicorn entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)