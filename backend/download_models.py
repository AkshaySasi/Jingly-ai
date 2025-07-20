import os
from huggingface_hub import snapshot_download
from bark import preload_models

def download_all_models():
    """
    Downloads and caches all required AI models locally.
    """
    # --- Download MusicGen ---
    musicgen_model_id = "facebook/musicgen-small"
    musicgen_local_dir = os.path.join("models", "musicgen-small")
    print(f"🎵 Downloading MusicGen model '{musicgen_model_id}'...")
    if not os.path.exists(musicgen_local_dir):
        os.makedirs(musicgen_local_dir)
        snapshot_download(
            repo_id=musicgen_model_id,
            local_dir=musicgen_local_dir,
            local_dir_use_symlinks=False,
        )
        print("✅ MusicGen model downloaded successfully!")
    else:
        print("✅ MusicGen model already exists.")

    # --- Download and preload Bark ---
    print("\n🎤 Downloading Bark models...")
    preload_models()
    print("✅ Bark models downloaded and cached successfully!")


if __name__ == "__main__":
    print("--- Starting AI Model Download Process ---")
    download_all_models()
    print("\n--- All models are ready! ---")