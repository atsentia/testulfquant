#!/usr/bin/env python3
"""
Download GPT-OSS-20B model from Hugging Face Hub.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm


def download_gpt_oss_20b(output_dir: str = "models/gpt-oss-20b"):
    """
    Download GPT-OSS-20B model from Hugging Face.
    
    Args:
        output_dir: Directory to save the model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading GPT-OSS-20B to {output_path}")
    print("Note: This model is approximately 42GB. Ensure you have sufficient disk space.")
    print("=" * 60)
    
    try:
        # Download model
        snapshot_download(
            repo_id="openai/gpt-oss-20b",
            local_dir=output_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"\n✓ Model downloaded successfully to {output_path}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in output_path.iterdir():
            size = file.stat().st_size / (1024**3)  # Convert to GB
            print(f"  - {file.name}: {size:.2f} GB")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (>50GB recommended)")
        print("3. Try running with: huggingface-cli login")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download GPT-OSS-20B model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/gpt-oss-20b",
        help="Directory to save the model"
    )
    
    args = parser.parse_args()
    download_gpt_oss_20b(args.output_dir)


if __name__ == "__main__":
    main()