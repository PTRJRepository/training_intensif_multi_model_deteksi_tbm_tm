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

# Force forward slashes in paths for Windows compatibility
os.environ['YOLO_PROJECT'] = str(Path(__file__).parent.parent / 'runs').replace('\\', '/')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Comprehensive patch for Windows path issues in Ultralytics
_original_open = open
_original_mkdir = Path.mkdir
_original_save = Path.write_bytes

def _patched_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    """Patch mkdir to handle Windows paths with backslashes."""
    try:
        return _original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)
    except OSError as e:
        if 'Invalid argument' in str(e) or 'invalid' in str(e).lower():
            # Try with forward slashes
            new_path = Path(str(self).replace('\\', '/'))
            return _original_mkdir(new_path, mode=mode, parents=parents, exist_ok=exist_ok)
        raise

Path.mkdir = _patched_mkdir

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

    # Build model
    print(f"\n[INFO] Initializing {model_cfg.model_name}...")
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

    # Start training
    print(f"\n[INFO] Starting training...")
    print(f"[INFO] Output: {project_cfg.save_dir / project_cfg.experiment_name}")
    print("=" * 60)

    # Train without checkpoint save issues by saving to temp dir, then copying
    train_args['project'] = str(project_cfg.save_dir).replace('\\', '/')
    train_args['name'] = str(project_cfg.experiment_name).replace('\\', '/')
    
    results = model.train(**train_args)

    print("\n" + "=" * 60)
    print("[SUCCESS] Training complete!")
    best_path = Path(project_cfg.save_dir) / project_cfg.experiment_name / "weights" / "best.pt"
    print(f"[INFO] Best model: {str(best_path).replace(chr(92), '/')}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
