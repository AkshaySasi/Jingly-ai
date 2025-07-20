import os
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load from local directory
model = MusicGen.get_pretrained("musicgen_model")
model.set_generation_params(duration=30)  

def generate_jingle(prompt, reference_audio_path=None, output_path="outputs/jingle_output1.wav"):
    print(f"🎼 Generating music for prompt:\n{prompt}\n")

    if reference_audio_path and os.path.exists(reference_audio_path):
        print(f"🎧 Using reference audio: {reference_audio_path}")
        wav, sr = torchaudio.load(reference_audio_path)
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=model.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
        wav = wav[:, :model.sample_rate * 30]
        output = model.generate_with_chroma([prompt], wav)[0]
    else:
        output = model.generate([prompt])[0]

    audio_write(output_path.replace(".wav", ""), output.cpu(), model.sample_rate, strategy="loudness")
    return output_path
