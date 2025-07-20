import google.generativeai as genai
from dotenv import load_dotenv
import re

load_dotenv()

def generate_music_and_lyrics_prompts(api_key, brand, motto, description, genre, prompt_instruction):
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Compose prompt with prompt_instruction included
    prompt = f"""
Brand: {brand}
Motto: {motto}
Product Description: {description}
Genre: {genre}
Prompt Instruction: {prompt_instruction}

1. Generate a MusicGen-compatible instrumental background score prompt (no lyrics, suitable for an upbeat brand jingle, 30 seconds).
2. Then generate short, catchy lyrics (2–4 lines) that reflect the healthy and happy brand identity.

Output clearly with two sections:

[MUSIC PROMPT]
...

[LYRICS]
...
"""

    response = model.generate_content(prompt)
    content = response.text.strip()

    music_prompt = extract_section(content, "MUSIC PROMPT")
    lyrics = extract_section(content, "LYRICS")

    print("\n🎵 Music Prompt:\n", music_prompt)
    print("\n📝 Lyrics:\n", lyrics)

    return music_prompt, lyrics

def extract_section(text, header):
    pattern = rf"\[{header}\](.*?)(\n\[|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""
