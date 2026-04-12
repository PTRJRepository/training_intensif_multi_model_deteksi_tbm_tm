"""
High Accuracy Training Launcher for TBM Detection
Fixed path issues and optimized settings for small objects.

Run this instead of Train/train.py for higher accuracy.

Usage:
    python _dev_utils/train_high_accuracy.py
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Run training with high accuracy settings."""

    print("=" * 70)
    print("TBM HIGH ACCURACY TRAINING")
    print("=" * 70)
    print("""
Settings optimized for:
- Small objects (~10px) - using 1280x1280 images
- Higher model capacity (yolo11s)
- Enhanced augmentation (copy_paste=0.2)
- More epochs (300)
    """)

    # Build the training command with forward slashes
    cmd = [
        sys.executable,
        "Train/train.py",
        "--model", "yolo11s",
        "--img-size", "1280",
        "--epochs", "300",
        "--batch", "2",
        "--copy-paste", "0.2",
        "--device", "cpu"
    ]

    print(f"[INFO] Command: {' '.join(cmd)}")
    print("=" * 70)

    # Run with forward-slash paths to avoid Windows OSError
    env = {
        "YOLO_CONFIG_DIR": str(PROJECT_ROOT / "_dev_utils" / ".yolo")
    }

    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
        return result.returncode
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
