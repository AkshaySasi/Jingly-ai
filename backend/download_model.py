from huggingface_hub import snapshot_download
import os

def download_musicgen_exclude_pytorch(save_dir="./musicgen_model"):
    unwanted_files = ["pytorch_model.bin"]

    # Download model snapshot excluding 'pytorch_model.bin'
    snapshot_download(
        repo_id="facebook/musicgen-small",
        local_dir=save_dir,
        local_dir_use_symlinks=False,  # Copy files instead of symlinking
        ignore_patterns=unwanted_files  # Exclude specific files
    )

    print(f"Model downloaded to '{save_dir}' without: {', '.join(unwanted_files)}")

if __name__ == "__main__":
    download_musicgen_exclude_pytorch()
