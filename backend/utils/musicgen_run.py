# utils/music_generator.py
import os
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_PATH = os.path.join("models", "musicgen-small")

def generate_music(prompt: str, duration: int, output_path: str) -> str:
    """
    Generates instrumental music using MusicGen.
    """
    print(f"🎵 Generating music for prompt: '{prompt}'...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    audio_values = model.generate(**inputs, max_new_tokens=int(duration * 50)) # ~50 tokens per second

    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_np = audio_values[0].cpu().numpy()
    
    sf.write(output_path, audio_np, sampling_rate)
    print(f"✅ Music track saved to: {output_path}")
    return output_path