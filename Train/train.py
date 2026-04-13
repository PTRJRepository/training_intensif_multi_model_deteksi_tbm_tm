"""
Main training script for TBM (Young Palm Tree) Detection.
Focus: High recall, high detection capability for small objects.

Usage:
    python Train/train.py
    python Train/train.py --model yolo11s --epochs 200 --batch 8
"""

import argparse
import sys
import os
import io
import shutil
from pathlib import Path
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Train.config import (
    get_config, print_config,
    ModelConfig, DatasetConfig, TrainingConfig,
    EvaluationConfig, ProjectConfig
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TBM Detection Model'
    )

    # Model
    parser.add_argument('--model', type=str, default='yolo11n',
                       help='Model architecture (yolov8n, yolov8s, yolo11n, yolo11s)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Pretrained weights path (default: ImageNet pretrained)')

    # Training
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=None,
                       help='Input image size')

    # Dataset
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data.yaml (default: Dataset/dataset_pakai_ini/data.yaml)')

    # Optimization
    parser.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer')

    # Augmentation emphasis
    parser.add_argument('--mosaic', type=float, default=None,
                       help='Mosaic augmentation (0.0-1.0)')
    parser.add_argument('--mixup', type=float, default=None,
                       help='MixUp augmentation (0.0-1.0)')
    parser.add_argument('--copy-paste', type=float, default=None,
                       help='Copy-paste augmentation (0.0-1.0)')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (0 for GPU, cpu for CPU)')

    # Output
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')

    return parser.parse_args()


def override_config(config: object, args: argparse.Namespace):
    """Override config with command line arguments."""
    for arg, value in vars(args).items():
        if value is not None and hasattr(config, arg):
            setattr(config, arg, value)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load default configs
    model_cfg, dataset_cfg, training_cfg, evaluation_cfg, project_cfg = get_config()

    # Override with command line arguments
    if args.model:
        model_cfg.model_name = args.model
    if args.weights:
        model_cfg.weights = args.weights
    if args.epochs:
        training_cfg.epochs = args.epochs
    if args.batch:
        training_cfg.batch = args.batch
    if args.img_size:
        dataset_cfg.img_size = args.img_size
    if args.data:
        dataset_cfg.data_yaml = Path(args.data)
    if args.lr:
        training_cfg.lr0 = args.lr
    if args.optimizer:
        training_cfg.optimizer = args.optimizer
    if args.mosaic:
        training_cfg.mosaic = args.mosaic
    if args.mixup:
        training_cfg.mixup = args.mixup
    if args.copy_paste:
        training_cfg.copy_paste = args.copy_paste
    if args.device:
        training_cfg.device = args.device
    if args.name:
        project_cfg.experiment_name = args.name
    if args.resume:
        training_cfg.resume = True

    # Print configuration
    print_config(model_cfg, dataset_cfg, training_cfg, evaluation_cfg)

    # Validate dataset path
    if not dataset_cfg.data_yaml.exists():
        print(f"\n[ERROR] Dataset not found: {dataset_cfg.data_yaml}")
        print("Please run preparation scripts first:")
        print("  python Tools/prepare_multi_image_dataset.py")
        sys.exit(1)

    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics not installed.")
        print("Install with: pip install ultralytics")
        sys.exit(1)

    # Build model - train from scratch (not fine-tuning)
    print(f"\n[INFO] Initializing {model_cfg.model_name} from scratch...")
    model = YOLO(f'{model_cfg.model_name}.pt')

    # Training arguments optimized for small object detection
    train_args = {
        # Dataset
        'data': str(dataset_cfg.data_yaml),
        'imgsz': dataset_cfg.img_size,
        'cache': dataset_cfg.cache_images,

        # Training
        'epochs': training_cfg.epochs,
        'batch': training_cfg.batch,
        'device': training_cfg.device,
        'workers': training_cfg.workers,
        'optimizer': training_cfg.optimizer,
        'lr0': training_cfg.lr0,
        'lrf': training_cfg.lrf,
        'momentum': training_cfg.momentum,
        'weight_decay': training_cfg.weight_decay,
        'patience': training_cfg.patience,
        'save_period': training_cfg.epochs,  # Only save at the end to avoid Windows OSError
        'resume': training_cfg.resume,

        # Augmentation - CRITICAL for small objects
        'hsv_h': training_cfg.hsv_h,
        'hsv_s': training_cfg.hsv_s,
        'hsv_v': training_cfg.hsv_v,
        'degrees': training_cfg.degrees,
        'translate': training_cfg.translate,
        'scale': training_cfg.scale,
        'shear': training_cfg.shear,
        'perspective': training_cfg.perspective,
        'flipud': training_cfg.flipud,
        'fliplr': training_cfg.fliplr,
        'mosaic': training_cfg.mosaic,
        'mixup': training_cfg.mixup,
        'copy_paste': training_cfg.copy_paste,
        'close_mosaic': training_cfg.close_mosaic,

        # Output - force forward slashes for Windows path compatibility
        'project': str(project_cfg.save_dir).replace('\\', '/'),
        'name': str(project_cfg.experiment_name).replace('\\', '/'),
        'exist_ok': project_cfg.exist_ok,

        # Verbose
        'verbose': True,
    }

    # Use short temp path for training to avoid Windows MAX_PATH issue
    # Then copy results back to original location
    original_save_dir = Path(project_cfg.save_dir)
    original_exp_name = str(project_cfg.experiment_name).replace('\\', '/')

    # Short training path: D:/tmp_train_2 (avoids Windows 260 char limit)
    short_train_dir = Path('D:/tmp_train_2')
    short_train_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Starting training...")
    print(f"[INFO] Training dir (temp): {short_train_dir}")
    print(f"[INFO] Final output: {original_save_dir / original_exp_name}")
    print("=" * 60)

    # Train with short path
    train_args['project'] = str(short_train_dir).replace('\\', '/')
    train_args['name'] = str(project_cfg.experiment_name).replace('\\', '/')
    train_args['save_period'] = -1  # Disable saves during training
    train_args['patience'] = 0  # Disable early stopping - FORCE FULL TRAINING

    try:
        results = model.train(**train_args)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        # Cleanup on error
        try:
            shutil.rmtree(short_train_dir)
        except Exception:
            pass
        raise

    # Manual save after training completes using YOLO's native save method
    weights_dir = short_train_dir / original_exp_name / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Save best and last model
    best_path = weights_dir / 'best.pt'
    last_path = weights_dir / 'last.pt'

    print(f"\n[INFO] Saving model...")
    try:
        model.save(str(last_path))
        model.save(str(best_path))
        print(f"[INFO] Model saved successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        # Try alternative save method
        print(f"[INFO] Trying alternative save method...")
        import torch
        torch.save({'model': model.model}, str(best_path))
        torch.save({'model': model.model}, str(last_path))
        print(f"[INFO] Model saved with alternative method!")

    # Copy results back to original location
    src_dir = short_train_dir / original_exp_name
    dst_dir = original_save_dir / original_exp_name

    if src_dir.exists():
        # Remove old results if exist
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        print(f"\n[INFO] Results copied to: {dst_dir}")

        # Cleanup temp
        try:
            shutil.rmtree(short_train_dir)
        except Exception:
            pass
    else:
        print(f"\n[WARNING] Training results not found at: {src_dir}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Training complete!")
    best_path = dst_dir / "weights" / "best.pt"
    print(f"[INFO] Best model: {str(best_path).replace(chr(92), '/')}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
