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
import datetime
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Train.config import (
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    get_config,
    print_config,
)
from Train.runtime import print_runtime_summary, resolve_training_runtime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TBM Detection Model")

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n",
        help="Model architecture (yolov8n, yolov8s, yolo11n, yolo11s)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Pretrained weights path (default: ImageNet pretrained)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--img-size", type=int, default=None, help="Input image size")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data.yaml (default: Dataset/dataset_pakai_ini/data.yaml)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer",
    )
    parser.add_argument(
        "--mosaic", type=float, default=None, help="Mosaic augmentation (0.0-1.0)"
    )
    parser.add_argument(
        "--mixup", type=float, default=None, help="MixUp augmentation (0.0-1.0)"
    )
    parser.add_argument(
        "--copy-paste",
        type=float,
        default=None,
        help="Copy-paste augmentation (0.0-1.0)",
    )
    parser.add_argument(
        "--zoom-in",
        type=float,
        default=None,
        help="Zoom in factor for scale augmentation (e.g., 1.5)",
    )
    parser.add_argument(
        "--zoom-out",
        type=float,
        default=None,
        help="Zoom out factor for scale augmentation (e.g., 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, 0 for GPU, cpu for CPU)",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Data loader workers"
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Force-enable mixed precision",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Force-disable mixed precision",
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.set_defaults(amp=None)

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

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        errors.append("Empty or invalid YAML file")
        return False, errors

    base_dir = yaml_path.parent
    paths_to_check = {}

    for key in ["train", "val"]:
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
            "names": data.get("names", {0: "TBM"}),
            "train": str(train_path),
            "val": str(val_path),
        }

        with open(yaml_path, "w") as f:
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
    if args.epochs is not None:
        training_cfg.epochs = args.epochs
    if args.batch is not None:
        training_cfg.batch = args.batch
    if args.img_size is not None:
        dataset_cfg.img_size = args.img_size
    if args.data:
        dataset_cfg.data_yaml = Path(args.data)
    if args.lr is not None:
        training_cfg.lr0 = args.lr
    if args.optimizer:
        training_cfg.optimizer = args.optimizer
    if args.mosaic is not None:
        training_cfg.mosaic = args.mosaic
    if args.mixup is not None:
        training_cfg.mixup = args.mixup
    if args.copy_paste is not None:
        training_cfg.copy_paste = args.copy_paste
    if args.zoom_in is not None:
        training_cfg.zoom_in_factor = args.zoom_in
    if args.zoom_out is not None:
        training_cfg.zoom_out_factor = args.zoom_out
    if args.device:
        training_cfg.device = args.device
    if args.workers is not None:
        training_cfg.workers = args.workers
    if args.amp is not None:
        training_cfg.amp = args.amp
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

    runtime = resolve_training_runtime(
        requested_device=training_cfg.device,
        requested_amp=training_cfg.amp,
        requested_workers=training_cfg.workers,
    )
    training_cfg.device = runtime.resolved_device
    training_cfg.amp = runtime.amp
    training_cfg.workers = runtime.workers

    print_runtime_summary(runtime)
    print_config(model_cfg, dataset_cfg, training_cfg, evaluation_cfg)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics not installed.")
        print("Install with: pip install ultralytics")
        sys.exit(1)

    print(f"\n[INFO] Initializing {model_cfg.model_name} from scratch...")
    model = YOLO(f"{model_cfg.model_name}.pt")

    train_args = {
        "data": str(dataset_cfg.data_yaml),
        "imgsz": dataset_cfg.img_size,
        "cache": dataset_cfg.cache_images,
        "epochs": training_cfg.epochs,
        "batch": training_cfg.batch,
        "device": training_cfg.device,
        "workers": training_cfg.workers,
        "optimizer": training_cfg.optimizer,
        "lr0": training_cfg.lr0,
        "lrf": training_cfg.lrf,
        "momentum": training_cfg.momentum,
        "weight_decay": training_cfg.weight_decay,
        "patience": training_cfg.patience,
        "save_period": training_cfg.epochs,
        "resume": training_cfg.resume,
        "box": training_cfg.box,
        "cls": training_cfg.cls,
        "dfl": training_cfg.dfl,
        "hsv_h": training_cfg.hsv_h,
        "hsv_s": training_cfg.hsv_s,
        "hsv_v": training_cfg.hsv_v,
        "degrees": training_cfg.degrees,
        "translate": training_cfg.translate,
        "scale": training_cfg.scale,
        "shear": training_cfg.shear,
        "perspective": training_cfg.perspective,
        "flipud": training_cfg.flipud,
        "fliplr": training_cfg.fliplr,
        "mosaic": training_cfg.mosaic,
        "mixup": training_cfg.mixup,
        "copy_paste": training_cfg.copy_paste,
        "close_mosaic": training_cfg.close_mosaic,
        "erasing": training_cfg.erasing,
        "project": str(project_cfg.save_dir).replace("\\", "/"),
        "name": str(project_cfg.experiment_name).replace("\\", "/"),
        "exist_ok": project_cfg.exist_ok,
        "verbose": True,
        "warmup_epochs": 3.0,
        "warmup_bias_lr": 0.0001,
        "amp": training_cfg.amp,
    }

    original_save_dir = Path(project_cfg.save_dir)
    original_exp_name = str(project_cfg.experiment_name).replace("\\", "/")

    short_train_dir = Path("D:/tmp_train_2")
    short_train_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Starting training...")
    print(f"[INFO] Training dir (temp): {short_train_dir}")
    print(f"[INFO] Final output: {original_save_dir / original_exp_name}")
    print("=" * 60)

    train_args["project"] = str(short_train_dir).replace("\\", "/")
    train_args["name"] = str(project_cfg.experiment_name).replace("\\", "/")
    train_args["save_period"] = -1
    train_args["patience"] = 0

    try:
        results = model.train(**train_args)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        try:
            shutil.rmtree(short_train_dir)
        except Exception:
            pass
        raise

    weights_dir = short_train_dir / original_exp_name / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    print(f"\n[INFO] Saving model...")
    try:
        model.save(str(last_path))
        model.save(str(best_path))
        print(f"[INFO] Model saved successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        import torch

        torch.save({"model": model.model}, str(best_path))
        torch.save({"model": model.model}, str(last_path))
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

    generate_documentation(
        dst_dir, model_cfg, dataset_cfg, training_cfg, evaluation_cfg, project_cfg
    )

    return results


def generate_documentation(
    dst_dir, model_cfg, dataset_cfg, training_cfg, evaluation_cfg, project_cfg
):
    """Generate README.md documentation for the trained model."""
    import datetime

    readme_content = f"""# Training Results Documentation

## Experiment: {project_cfg.experiment_name}
**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Model Configuration
| Parameter | Value |
|-----------|-------|
| Architecture | {model_cfg.model_name} |
| Image Size | {dataset_cfg.img_size} |
| Classes | {", ".join(dataset_cfg.class_names)} |

---

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | {training_cfg.epochs} |
| Batch Size | {training_cfg.batch} |
| Optimizer | {training_cfg.optimizer} |
| Learning Rate | {training_cfg.lr0} |
| Final LR Factor | {training_cfg.lrf} |
| Momentum | {training_cfg.momentum} |
| Weight Decay | {training_cfg.weight_decay} |

---

## Augmentations Used
This model was trained with extensive augmentations to handle various conditions including grayscale, zoom, and shape variations:

| Augmentation | Value | Purpose |
|--------------|-------|---------|
| HSV Hue | {training_cfg.hsv_h} | Color variation (grayscale-like learning) |
| HSV Saturation | {training_cfg.hsv_s} | High saturation range |
| HSV Value | {training_cfg.hsv_v} | Brightness variation |
| Degrees | {training_cfg.degrees} | Rotation for shape recognition |
| Translate | {training_cfg.translate} | Position variation |
| Scale | {training_cfg.scale} | Zoom in/out (0.1x - 1.9x) |
| Shear | {training_cfg.shear} | Shape deformation |
| Perspective | {training_cfg.perspective} | 3D-like variation |
| Flip Up/Down | {training_cfg.flipud} | Vertical flip |
| Flip Left/Right | {training_cfg.fliplr} | Horizontal flip |
| Mosaic | {training_cfg.mosaic} | Context learning |
| MixUp | {training_cfg.mixup} | Blended learning |
| Copy-Paste | {training_cfg.copy_paste} | Object duplication |
| Random Erasing | {training_cfg.erasing} | Occlusion handling |

---

## Loss Weights
| Component | Weight |
|-----------|--------|
| Box Loss | {training_cfg.box} |
| Class Loss | {training_cfg.cls} |
| DFL Loss | {training_cfg.dfl} |

---

## Recommended Inference Settings

### Basic Inference
```bash
python Train/inference.py --weights "{dst_dir}/weights/best.pt" --source <image_path>
```

### High Recall Inference (Recommended for Small Objects)
```bash
python Train/inference.py --weights "{dst_dir}/weights/best.pt" --source <image_path> --conf 0.001 --iou 0.3 --augment
```

### Multi-Scale Inference with Ensemble
```bash
python Train/inference.py --weights "{dst_dir}/weights/best.pt" --source <image_path> --conf 0.001 --iou 0.3 --augment --img-size 640
```

### Sliding Window Inference (for large images)
Use window size 640x640 with 20% overlap for optimal small object detection.

---

## Evaluation Metrics Focus
| Metric | Threshold |
|--------|-----------|
| IoU Threshold | {evaluation_cfg.iou_thres} (low for higher recall) |
| Confidence Threshold | {evaluation_cfg.conf_thres} (very low for high recall) |
| Max Detections | {evaluation_cfg.max_det} per image |

---

## Key Features for Inference

1. **Grayscale/Low Light**: Model trained with high HSV variation - performs well on desaturated images
2. **Zoom Handling**: Scale augmentation (0.1x - 1.9x) - use multi-scale inference
3. **Shape Recognition**: High rotation ({training_cfg.degrees}°) and shear ({training_cfg.shear}) - robust to tree shapes
4. **Small Objects**: Optimized for objects as small as ~10-20 pixels

---

## Files in This Directory
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- `results.png` - Training metrics plots
- `results.csv` - Training metrics data
- `README.md` - This documentation

---

## Training Hardware
- Device: {training_cfg.device}
- Workers: {training_cfg.workers}

---
*Generated automatically by train.py*
"""

    readme_path = dst_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"\n[INFO] Documentation saved to: {readme_path}")


if __name__ == "__main__":
    main()
