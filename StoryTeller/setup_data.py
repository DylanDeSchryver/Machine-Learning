#!/usr/bin/env python3
import sys
import subprocess

print("=" * 70)
print("STORYTELLER DATA SETUP")
print("=" * 70)

# Step 1: Download
print("\nStep 1/3: Downloading TinyStories...")
subprocess.run([sys.executable, "src/storyteller/data/download_stories.py", 
                "--output_dir", "data/raw", "--datasets", "tinystories"], check=True)

# Step 2: Preprocess
print("\nStep 2/3: Preprocessing...")
subprocess.run([sys.executable, "src/storyteller/data/preprocess.py",
                "--input_dir", "data/raw", "--output_dir", "data/processed"], check=True)

# Step 3: Tokenizer
print("\nStep 3/3: Training tokenizer...")
subprocess.run([sys.executable, "src/storyteller/data/tokenizer_training.py",
                "--input_file", "data/processed/train.txt",
                "--output_dir", "data/tokenizers/storyteller-tokenizer",
                "--vocab_size", "50000"], check=True)

print("\n✓ COMPLETE! Now run training:")
print("  python src/storyteller/training/train.py --config configs/moe_advanced.yaml")
