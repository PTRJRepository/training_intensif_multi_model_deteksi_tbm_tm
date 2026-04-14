"""
Main training script for TBM (Young Palm Tree) Detection.
Focus: High recall, high detection capability for small objects.

Robust features:
- Auto-validates and fixes data.yaml paths
- Uses absolute paths throughout

Usage:
    python Train/train.py
    python Train/train.py --model yolo11s --epochs 200 --batch 8
"""

import argparse
import sys
import os
import io
import shutil
import yaml
from pathlib import Path
import tempfile

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

    parser.add_argument('--model', type=str, default='yolo11n',
                       help='Model architecture (yolov8n, yolov8s, yolo11n, yolo11s)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Pretrained weights path (default: ImageNet pretrained)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=None,
                       help='Input image size')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data.yaml (default: Dataset/dataset_pakai_ini/data.yaml)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer')
    parser.add_argument('--mosaic', type=float, default=None,
                       help='Mosaic augmentation (0.0-1.0)')
    parser.add_argument('--mixup', type=float, default=None,
                       help='MixUp augmentation (0.0-1.0)')
    parser.add_argument('--copy-paste', type=float, default=None,
                       help='Copy-paste augmentation (0.0-1.0)')
    parser.add_argument('--zoom-in', type=float, default=None,
                       help='Zoom in factor for scale augmentation (e.g., 1.5)')
    parser.add_argument('--zoom-out', type=float, default=None,
                       help='Zoom out factor for scale augmentation (e.g., 0.5)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (0 for GPU, cpu for CPU)')
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


def validate_and_fix_data_yaml(yaml_path):
    """Validate and fix data.yaml with robust path handling."""
    print(f"\n[INFO] Validating data.yaml: {yaml_path}")
    
    errors = []
    
    if not yaml_path.exists():
        errors.append(f"File not found: {yaml_path}")
        return False, errors
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        errors.append("Empty or invalid YAML file")
        return False, errors
    
    base_dir = yaml_path.parent
    paths_to_check = {}
    
    for key in ['train', 'val']:
        if key in data:
            path_str = data[key]
            if Path(path_str).is_absolute():
                paths_to_check[key] = Path(path_str)
            else:
                paths_to_check[key] = base_dir / path_str
    
    for key, path in paths_to_check.items():
        if not path.exists():
            errors.append(f"Missing {key} directory: {path}")
    
    if errors:
        print(f"[WARNING] data.yaml has issues:")
        for err in errors:
            print(f"  - {err}")
        print("\n[INFO] Fixing data.yaml with absolute paths...")
        
        train_path = base_dir / "images" / "train"
        val_path = base_dir / "images" / "val"
        
        if not train_path.exists():
            errors.append(f"Train path does not exist: {train_path}")
        if not val_path.exists():
            errors.append(f"Val path does not exist: {val_path}")
        
        if errors:
            return False, errors
        
        fixed_data = {
            'names': data.get('names', {0: 'TBM'}),
            'train': str(train_path),
            'val': str(val_path)
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(fixed_data, f, default_flow_style=False)
        
        print(f"[INFO] data.yaml fixed!")
        print(f"  train: {train_path}")
        print(f"  val: {val_path}")
        return True, []
    
    print(f"[INFO] data.yaml is valid!")
    return True, []


def main():
    """Main training function."""
    args = parse_args()

    model_cfg, dataset_cfg, training_cfg, evaluation_cfg, project_cfg = get_config()

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
    if args.zoom_in:
        training_cfg.zoom_in_factor = args.zoom_in
    if args.zoom_out:
        training_cfg.zoom_out_factor = args.zoom_out
    if args.device:
        training_cfg.device = args.device
    if args.name:
        project_cfg.experiment_name = args.name
    if args.resume:
        training_cfg.resume = True

    data_yaml_default = PROJECT_ROOT / "Dataset" / "dataset_pakai_ini" / "data.yaml"
    if not dataset_cfg.data_yaml or not dataset_cfg.data_yaml.exists():
        dataset_cfg.data_yaml = data_yaml_default

    valid, errors = validate_and_fix_data_yaml(dataset_cfg.data_yaml)
    if not valid:
        print(f"\n[ERROR] Cannot proceed with invalid dataset:")
        for err in errors:
            print(f"  - {err}")
        print("Please run preparation scripts first:")
        print("  python Tools/prepare_multi_image_dataset.py")
        sys.exit(1)

    print_config(model_cfg, dataset_cfg, training_cfg, evaluation_cfg)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics not installed.")
        print("Install with: pip install ultralytics")
        sys.exit(1)

    print(f"\n[INFO] Initializing {model_cfg.model_name} from scratch...")
    model = YOLO(f'{model_cfg.model_name}.pt')

    train_args = {
        'data': str(dataset_cfg.data_yaml),
        'imgsz': dataset_cfg.img_size,
        'cache': dataset_cfg.cache_images,
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
        'save_period': training_cfg.epochs,
        'resume': training_cfg.resume,
        'box': training_cfg.box,
        'cls': training_cfg.cls,
        'dfl': training_cfg.dfl,
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
        'erasing': 0.4,
        'crop_fraction': 1.0,
        'project': str(project_cfg.save_dir).replace('\\', '/'),
        'name': str(project_cfg.experiment_name).replace('\\', '/'),
        'exist_ok': project_cfg.exist_ok,
        'verbose': True,
        'warmup_epochs': 3.0,
        'warmup_bias_lr': 0.0001,
    }

    original_save_dir = Path(project_cfg.save_dir)
    original_exp_name = str(project_cfg.experiment_name).replace('\\', '/')

    short_train_dir = Path('D:/tmp_train_2')
    short_train_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Starting training...")
    print(f"[INFO] Training dir (temp): {short_train_dir}")
    print(f"[INFO] Final output: {original_save_dir / original_exp_name}")
    print("=" * 60)

    train_args['project'] = str(short_train_dir).replace('\\', '/')
    train_args['name'] = str(project_cfg.experiment_name).replace('\\', '/')
    train_args['save_period'] = -1
    train_args['patience'] = 0

    try:
        results = model.train(**train_args)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        try:
            shutil.rmtree(short_train_dir)
        except Exception:
            pass
        raise

    weights_dir = short_train_dir / original_exp_name / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_path = weights_dir / 'best.pt'
    last_path = weights_dir / 'last.pt'

    print(f"\n[INFO] Saving model...")
    try:
        model.save(str(last_path))
        model.save(str(best_path))
        print(f"[INFO] Model saved successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        import torch
        torch.save({'model': model.model}, str(best_path))
        torch.save({'model': model.model}, str(last_path))
        print(f"[INFO] Model saved with alternative method!")

    src_dir = short_train_dir / original_exp_name
    dst_dir = original_save_dir / original_exp_name

    if src_dir.exists():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        print(f"\n[INFO] Results copied to: {dst_dir}")

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