#!/usr/bin/env python3
"""
Vast.ai training script for ai-t9.

Handles downloading/uploading from Cloudflare R2, checkpointing, and graceful shutdown.
"""

import os
import signal
import sys
from pathlib import Path
import boto3
from botocore.config import Config
import subprocess
import time

# R2 configuration from environment
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')
R2_ENDPOINT = os.getenv('R2_ENDPOINT')
R2_BUCKET = os.getenv('R2_BUCKET', 'ai-t9-data')

# Local paths
DATA_DIR = Path('/data')
VOCAB_PATH = DATA_DIR / 'vocab.json'
DICT_PATH = DATA_DIR / 'dict.json'
PAIRS_PATH = DATA_DIR / 'pairs.npz'
CHECKPOINT_PATH = DATA_DIR / 'checkpoint.pth'
MODEL_PATH = DATA_DIR / 'model.npz'

# R2 keys
R2_VOCAB = 'vocab.json'
R2_DICT = 'dict.json'
R2_PAIRS = 'pairs.npz'
R2_CHECKPOINT = 'checkpoint.pth'
R2_MODEL = 'model.npz'

def get_r2_client():
    return boto3.client(
        's3',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        endpoint_url=R2_ENDPOINT,
        config=Config(signature_version='s3v4')
    )

def download_from_r2(r2_key, local_path):
    """Download file from R2 if it exists."""
    if local_path.exists():
        print(f"{local_path} already exists, skipping download")
        return
    try:
        client = get_r2_client()
        client.download_file(R2_BUCKET, r2_key, str(local_path))
        print(f"Downloaded {r2_key} to {local_path}")
    except client.exceptions.NoSuchKey:
        print(f"{r2_key} not found in R2")
    except Exception as e:
        print(f"Error downloading {r2_key}: {e}")

def upload_to_r2(local_path, r2_key):
    """Upload file to R2."""
    try:
        client = get_r2_client()
        client.upload_file(str(local_path), R2_BUCKET, r2_key)
        print(f"Uploaded {local_path} to {r2_key}")
    except Exception as e:
        print(f"Error uploading {local_path}: {e}")

def download_data():
    """Download required data files from R2."""
    print("Downloading data from R2...")
    download_from_r2(R2_VOCAB, VOCAB_PATH)
    download_from_r2(R2_DICT, DICT_PATH)
    download_from_r2(R2_PAIRS, PAIRS_PATH)
    download_from_r2(R2_CHECKPOINT, CHECKPOINT_PATH)

def upload_checkpoint():
    """Upload checkpoint and model to R2."""
    if CHECKPOINT_PATH.exists():
        upload_to_r2(CHECKPOINT_PATH, R2_CHECKPOINT)
    if MODEL_PATH.exists():
        upload_to_r2(MODEL_PATH, R2_MODEL)

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"Received signal {signum}, saving checkpoint...")
    upload_checkpoint()
    sys.exit(0)

def main():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # Download data
    download_data()

    # Check if required files exist
    if not VOCAB_PATH.exists():
        print("ERROR: vocab.json not found")
        sys.exit(1)
    if not PAIRS_PATH.exists():
        print("ERROR: pairs.npz not found")
        sys.exit(1)

    # Run training
    cmd = [
        'ai-t9-train',
        '--vocab', str(VOCAB_PATH),
        '--load-pairs', str(PAIRS_PATH),
        '--output', str(MODEL_PATH),
        '--checkpoint', str(CHECKPOINT_PATH),
        '--epochs', '1000',  # High number, will be interrupted
        '--device', 'auto',
        '--verbose'
    ]

    print("Starting training...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # Upload final checkpoint
        upload_checkpoint()

if __name__ == '__main__':
    main()