import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from bark import generate_audio, preload_models
from pydub import AudioSegment
import re
import torch
from scipy.io.wavfile import write as write_wav
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil
import threading
import time
from functools import wraps

# Configure logging
def setup_logging(log_file: Path):
    """Set up logging with file and console handlers."""
    try:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.handlers = []
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
BARK_CACHE_DIR = Path.home() / ".cache" / "suno" / "bark_v0"

# Ensure directories exist
try:
    for directory in [LOG_DIR, MODELS_DIR, OUTPUT_DIR, BARK_CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"Error creating directories: {str(e)}")
    raise

# Initialize logger
logger = setup_logging(LOG_DIR / "jingle_generator.log")

# Check CUDA availability
if not torch.cuda.is_available():
    logger.warning(
        "CUDA is not available. Using CPU, which may result in slower performance (vocal generation may take ~10-20 minutes). "
        "Consider installing PyTorch with CUDA support (see https://pytorch.org/get-started/locally/)."
    )

# Check disk space
def check_disk_space(path: Path, required_mb: float) -> bool:
    """Check if there is enough disk space at the given path."""
    try:
        total, used, free = shutil.disk_usage(path)
        free_mb = free / (1024 * 1024)  # Convert bytes to MB
        if free_mb < required_mb:
            logger.error(f"Insufficient disk space. Required: {required_mb} MB, Available: {free_mb:.2f} MB at {path}")
            return False
        logger.info(f"Disk space check passed. Available: {free_mb:.2f} MB at {path}")
        return True
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        raise

# Timeout decorator for Windows
def timeout(seconds):
    def decorator(func):
        def _handle_timeout():
            raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if exception[0] is not None:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in .env file")
    raise ValueError("GEMINI_API_KEY is required")

# Constants
BRAND_NAME = "Krawings"
MOTTO = "Crave Healthy, Live Happy"
PRODUCT_BRIEF = "Krawings is a healthy snack brand offering a range of nutritious and delicious snacks."
GENRE = "Upbeat Pop"
PROMPT_INSTRUCTION = (
    "Generate an instrumental music prompt (no lyrics) that reflects a healthy and happy brand vibe, "
    "suitable for a 30-second jingle."
)

def ensure_bark_model():
    """Check for local Bark model and download if not present."""
    try:
        bark_model_id = "suno/bark-small"
        bark_dir = MODELS_DIR / "bark-small"
        required_files = ["text_2.pt", "coarse_2.pt", "fine_2.pt"]
        
        # Check local model directory
        if bark_dir.exists():
            missing_files = [f for f in required_files if not (bark_dir / f).exists()]
            if not missing_files:
                logger.info(f"Bark-Small model found at {bark_dir}")
                # Copy to Bark cache
                for file in required_files:
                    src = bark_dir / file
                    dst = BARK_CACHE_DIR / file
                    if not dst.exists():
                        logger.info(f"Copying {file} to {BARK_CACHE_DIR}")
                        if not check_disk_space(BARK_CACHE_DIR, 4000):  # Assume ~4GB
                            raise ValueError("Not enough disk space to copy Bark model files")
                        shutil.copy2(src, dst)
                logger.info(f"Bark model files verified in cache at {BARK_CACHE_DIR}")
                return str(bark_dir)
            else:
                logger.warning(f"Local Bark model incomplete. Missing files: {', '.join(missing_files)}")
        
        # Download model if not present or incomplete
        logger.info("Downloading Bark-Small model...")
        if not check_disk_space(MODELS_DIR, 4000):  # Assume ~4GB
            raise ValueError("Not enough disk space to download Bark-Small model")
        bark_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=bark_model_id,
            local_dir=bark_dir,
            local_dir_use_symlinks=False
        )
        # Copy downloaded files to cache
        for file in required_files:
            src = bark_dir / file
            dst = BARK_CACHE_DIR / file
            if src.exists() and not dst.exists():
                logger.info(f"Copying {file} to {BARK_CACHE_DIR}")
                if not check_disk_space(BARK_CACHE_DIR, 4000):
                    raise ValueError("Not enough disk space to copy Bark model files")
                shutil.copy2(src, dst)
        logger.info("Bark-Small model downloaded and cached successfully")
        return str(bark_dir)
    except Exception as e:
        logger.error(f"Error ensuring Bark model: {str(e)}")
        raise

def download_models():
    """Download or verify MusicGen and Bark models."""
    try:
        # MusicGen model
        musicgen_model_id = "facebook/musicgen-small"
        musicgen_dir = MODELS_DIR / "musicgen-small"
        if not musicgen_dir.exists():
            logger.info("Downloading MusicGen model...")
            if not check_disk_space(MODELS_DIR, 2000):  # Assume ~2GB
                raise ValueError("Not enough disk space to download MusicGen model")
            musicgen_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=musicgen_model_id,
                local_dir=musicgen_dir,
                local_dir_use_symlinks=False
            )
            logger.info("MusicGen model downloaded successfully")
        else:
            logger.info("MusicGen model already available")

        # Bark-Small model
        ensure_bark_model()
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        raise

def generate_music_and_lyrics_prompts(api_key: str, brand: str, motto: str, description: str, genre: str, prompt_instruction: str) -> tuple[str, str]:
    """Generate music prompt and lyrics using Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

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
        if not response or not response.text:
            raise ValueError("No response from Gemini API")
        
        content = response.text.strip()
        music_prompt = extract_section(content, "MUSIC PROMPT")
        lyrics = extract_section(content, "LYRICS")

        if not music_prompt or not lyrics:
            raise ValueError("Failed to extract music prompt or lyrics from response")

        logger.info("Music Prompt:\n%s", music_prompt)
        logger.info("Lyrics:\n%s", lyrics)
        return music_prompt, lyrics
    except Exception as e:
        logger.error(f"Error generating prompts: {str(e)}")
        raise

def extract_section(text: str, header: str) -> str:
    """Extract a section from text based on header."""
    try:
        pattern = rf"\[{header}\](.*?)(\n\[|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    except Exception as e:
        logger.error(f"Error extracting section {header}: {str(e)}")
        return ""

def generate_jingle(prompt: str, output_path: Path = OUTPUT_DIR / "jingle_music.wav") -> str:
    """Generate background music using MusicGen."""
    try:
        if not check_disk_space(OUTPUT_DIR, 100):  # Assume ~100MB
            raise ValueError("Not enough disk space to generate music file")
        logger.info("Generating music for prompt:\n%s", prompt)
        model = MusicGen.get_pretrained(str(MODELS_DIR / "musicgen-small"))
        model.set_generation_params(duration=30)
        output = model.generate([prompt])[0]
        audio_write(
            str(output_path.with_suffix("")),
            output.cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        logger.info("Music generated at: %s", output_path)
        return str(output_path)
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}")
        raise

@timeout(seconds=1800) 
def generate_vocals(text: str, output_path: Path = OUTPUT_DIR / "jingle_voice.wav") -> str:
    """Generate vocals using Bark model."""
    try:
        if not check_disk_space(OUTPUT_DIR, 100):  # Assume ~100MB
            raise ValueError("Not enough disk space to generate vocals file")
        logger.info("Generating vocals for lyrics:\n%s", text)
        
        # Ensure Bark model files are in cache
        ensure_bark_model()
        
        # Load Bark models
        preload_models(
            text_use_gpu=False,
            coarse_use_gpu=False,
            fine_use_gpu=False,
            codec_use_gpu=False,
            force_reload=False
        )
        
        audio_array = generate_audio(text, history_prompt="v2/en_speaker_6")
        write_wav(output_path, rate=24000, data=audio_array)
        logger.info("Vocals generated at: %s", output_path)
        return str(output_path)
    except TimeoutError:
        logger.error("Vocal generation timed out after 10 minutes")
        raise
    except Exception as e:
        logger.error(f"Error generating vocals: {str(e)}")
        raise

def combine_audio(music_file: str, voice_file: str, output_file: Path = OUTPUT_DIR / "final_jingle.wav", music_reduction_db: int = 10) -> str:
    """Combine music and vocals into a final jingle."""
    try:
        if not check_disk_space(OUTPUT_DIR, 100):  # Assume ~100MB
            raise ValueError("Not enough disk space to generate final jingle")
        logger.info("Combining audio tracks...")
        music = AudioSegment.from_wav(music_file)
        voice = AudioSegment.from_wav(voice_file)
        quieter_music = music - music_reduction_db
        final_audio = quieter_music.overlay(voice)
        final_audio.export(output_file, format="wav")
        logger.info("Final jingle saved to: %s", output_file)
        return str(output_file)
    except Exception as e:
        logger.error(f"Error combining audio: {str(e)}")
        raise

def test_bark_speech():
    """Test Bark model with a sample text input."""
    try:
        logger.info("Testing Bark speech generation...")
        if not check_disk_space(OUTPUT_DIR, 100):  # Assume ~100MB
            raise ValueError("Not enough disk space to generate test speech")
        
        test_text = (
            "Hello, my name is Suno. And, uh — and I like pizza. [laughs] "
            "But I also have other interests such as playing tic tac toe."
        )
        output_path = OUTPUT_DIR / "test_speech.wav"
        generate_vocals(test_text, output_path)
        logger.info("Test speech generated successfully at: %s", output_path)
    except Exception as e:
        logger.error(f"Error testing Bark speech: {str(e)}")
        raise

def run_pipeline():
    """Run the jingle generation pipeline."""
    try:
        logger.info("Starting jingle generation pipeline...")

        # Check disk space
        if not check_disk_space(OUTPUT_DIR, 300):  # Assume ~300MB
            raise ValueError("Not enough disk space to run pipeline")
        if not check_disk_space(BARK_CACHE_DIR, 4000):  # Assume ~4GB
            raise ValueError("Not enough disk space for Bark model cache")

        # Download or verify models
        download_models()

        # Generate prompts
        logger.info("Building prompts using Gemini...")
        music_prompt, lyrics = generate_music_and_lyrics_prompts(
            GEMINI_API_KEY,
            BRAND_NAME,
            MOTTO,
            PRODUCT_BRIEF,
            GENRE,
            PROMPT_INSTRUCTION
        )

        # Define output paths
        music_file = OUTPUT_DIR / "jingle_music.wav"
        voice_file = OUTPUT_DIR / "jingle_voice.wav"
        final_output = OUTPUT_DIR / "final_jingle.wav"

        # Generate music
        logger.info("Generating background music...")
        generate_jingle(music_prompt, music_file)

        # Generate vocals
        logger.info("Generating vocals from lyrics...")
        generate_vocals(lyrics, voice_file)

        # Combine audio
        logger.info("Merging music and vocals...")
        combine_audio(music_file, voice_file, final_output)

        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the full pipeline
    run_pipeline()
    # Test Bark independently
    test_bark_speech()