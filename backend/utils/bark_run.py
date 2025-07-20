# utils/voice_generator.py
from bark import generate_audio, SAMPLE_RATE
import soundfile as sf

def generate_voice(text_prompt: str, output_path: str) -> str:
    """
    Generates a voice track with speech and sound effects using Bark.
    """
    print(f"🎤 Generating voice for text: '{text_prompt}'...")
    # The second return value is the audio numpy array
    _, audio_array = generate_audio(text_prompt, history_prompt="en_speaker_6") # Choose a speaker voice
    
    sf.write(output_path, audio_array, SAMPLE_RATE)
    print(f"✅ Voice track saved to: {output_path}")
    return output_path