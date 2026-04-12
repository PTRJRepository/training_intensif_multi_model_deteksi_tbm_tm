"""
High Accuracy Training Configuration for TBM Detection
Optimized for small object detection with maximum accuracy.

Usage:
    python Train/train.py --model yolo11s --img-size 1280 --batch 2 --epochs 300
    python _dev_utils/train_high_accuracy.py  # Run with this script
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_size: Literal['n', 's', 'm'] = 's'  # Changed from 'n' to 's' for better accuracy
    model_name: str = 'yolo11s'  # Changed from yolo11n


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    dataset_path: Path = Path(__file__).parent.parent / "Dataset" / "dataset_pakai_ini"
    data_yaml: Path = None

    class_names: List[str] = field(default_factory=lambda: ['tbm'])

    img_size: int = 1280  # Changed from 640 to 1280 - CRITICAL for small objects
    cache_images: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters - HIGH ACCURACY settings."""
    # Core settings
    epochs: int = 300  # Increased from 100
    batch: int = 2  # Reduced from 4 due to larger image size
    device: str = 'cpu'

    # Optimizer
    optimizer: Literal['SGD', 'Adam', 'AdamW'] = 'Adam'
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # Augmentation - MAXIMUM for small objects
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 15.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0  # Keep at maximum
    mixup: float = 0.1
    copy_paste: float = 0.2  # INCREASED from 0.1 - critical for small objects

    close_mosaic: int = 10

    # Other
    workers: int = 8
    patience: int = 100  # Increased - allow more time for improvement
    save_period: int = 10
    resume: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    iou_thres: float = 0.3
    conf_thres: float = 0.001
    max_det: int = 300

    eval_metrics: List[str] = field(default_factory=lambda: [
        'precision', 'recall', 'mAP50', 'mAP50-95'
    ])


@dataclass
class ProjectConfig:
    """Project/output configuration."""
    project_name: str = 'TBM_Detection'
    experiment_name: str = 'exp2_yolo11s_1280_high_acc'  # NEW experiment name
    save_dir: Path = Path(__file__).parent.parent / "Train" / "runs"
    exist_ok: bool = True


def get_config() -> tuple:
    """Get all configurations."""
    model = ModelConfig()
    dataset = DatasetConfig()
    training = TrainingConfig()
    evaluation = EvaluationConfig()
    project = ProjectConfig()

    return model, dataset, training, evaluation, project
