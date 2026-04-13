"""
Fine-tuning script for TBM Detection with high accuracy and recall.
Uses 640x640 detection window with optimizations for small objects.

Robust handling:
- Auto-validates and fixes data.yaml paths
- Uses absolute paths throughout
- Validates dataset before training

Usage:
    python Train/fine_tuning/train_ft.py
    python Train/fine_tuning/train_ft.py --epochs 150
"""

import argparse
import sys
import os
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Train.config import get_config, print_config


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune TBM Detection Model')
    parser.add_argument('--model', type=str, default='yolo11n',
                        help='Model architecture')
    parser.add_argument('--weights', type=str, default=None,
                       help='Pretrained weights path (default: train from scratch)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (640x640)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data.yaml')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate (lower for fine-tuning)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device')
    parser.add_argument('--name', type=str, default='ft_yolo11n_640_high_recall',
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    return parser.parse_args()


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
    args = parse_args()

    model_cfg, dataset_cfg, training_cfg, evaluation_cfg, project_cfg = get_config()

    model_cfg.model_name = args.model
    training_cfg.epochs = args.epochs
    training_cfg.batch = args.batch
    dataset_cfg.img_size = args.img_size
    training_cfg.lr0 = args.lr
    training_cfg.device = args.device
    project_cfg.experiment_name = args.name

    data_yaml_path = PROJECT_ROOT / "Dataset" / "dataset_pakai_ini" / "data.yaml"
    if args.data:
        dataset_cfg.data_yaml = Path(args.data)
    else:
        dataset_cfg.data_yaml = data_yaml_path

    if args.resume:
        training_cfg.resume = True

    valid, errors = validate_and_fix_data_yaml(dataset_cfg.data_yaml)
    if not valid:
        print(f"\n[ERROR] Cannot proceed with invalid dataset:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print_config(model_cfg, dataset_cfg, training_cfg, evaluation_cfg)

    if not dataset_cfg.data_yaml.exists():
        print(f"\n[ERROR] Dataset file not found: {dataset_cfg.data_yaml}")
        sys.exit(1)

    from ultralytics import YOLO

    if args.weights:
        print(f"\n[INFO] Loading base model: {args.weights}")
        model = YOLO(args.weights)
    else:
        print(f"\n[INFO] Training from scratch: {model_cfg.model_name}")
        model = YOLO(f'{model_cfg.model_name}.pt')

    train_args = {
        'data': str(dataset_cfg.data_yaml),
        'imgsz': dataset_cfg.img_size,
        'cache': dataset_cfg.cache_images,
        'epochs': training_cfg.epochs,
        'batch': training_cfg.batch,
        'device': training_cfg.device,
        'workers': training_cfg.workers,
        'optimizer': 'AdamW',
        'lr0': training_cfg.lr0,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'patience': 0,
        'save_period': -1,
        'resume': training_cfg.resume,
        'warmup_epochs': 3.0,
        'warmup_bias_lr': 0.0001,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.3,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.15,
        'close_mosaic': 10,
        'exist_ok': True,
        'verbose': True,
        'amp': False,
    }

    original_save_dir = PROJECT_ROOT / "Train" / "fine_tuning"
    original_exp_name = str(project_cfg.experiment_name).replace('\\', '/')

    short_train_dir = Path('D:/tmp_train_ft')
    short_train_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Starting fine-tuning...")
    print(f"[INFO] Base model: {args.weights}")
    print(f"[INFO] Image size: {args.img_size}x{args.img_size}")
    print(f"[INFO] Training dir (temp): {short_train_dir}")
    print(f"[INFO] Final output: {original_save_dir / original_exp_name}")
    print("=" * 60)

    train_args['project'] = str(short_train_dir).replace('\\', '/')
    train_args['name'] = str(project_cfg.experiment_name).replace('\\', '/')
    train_args['save_period'] = -1
    train_args['patience'] = 0

    import shutil
    
    try:
        results = model.train(**train_args)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        try:
            shutil.rmtree(short_train_dir)
        except Exception:
            pass
        raise

    weights_dir = short_train_dir / str(project_cfg.experiment_name).replace('\\', '/') / 'weights'
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

    src_dir = short_train_dir / str(project_cfg.experiment_name).replace('\\', '/')
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
    print("[SUCCESS] Fine-tuning complete!")
    best_path = dst_dir / "weights" / "best.pt"
    print(f"[INFO] Best model: {str(best_path).replace(chr(92), '/')}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()