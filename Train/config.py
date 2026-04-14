"""
Configuration for TBM (Young Palm Tree) Detection Training.
Focus: High recall, high detection capability for small objects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_size: Literal['n', 's', 'm'] = 's'
    model_name: str = 'yolo11s'  # yolo11s - better for small/many objects than yolo11n


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    # Base dataset path
    dataset_path: Path = Path(__file__).parent.parent / "Dataset" / "dataset_pakai_ini"
    data_yaml: Path = None  # Auto-detected if None

    # Class names
    class_names: List[str] = field(default_factory=lambda: ['tbm'])

    # Image settings
    img_size: int = 640  # Training image size (multiple of 32)
    cache_images: bool = True  # Cache images in RAM for faster training (enough RAM available)

    def __post_init__(self):
        if self.data_yaml is None:
            self.data_yaml = self.dataset_path / "data.yaml"


@dataclass
class TrainingConfig:
    """Training hyperparameters - optimized for small object detection."""
    epochs: int = 350
    batch: int = 4
    device: str = 'cpu'

    optimizer: Literal['SGD', 'Adam', 'AdamW'] = 'AdamW'
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 15.0
    translate: float = 0.1
    scale: float = 0.9
    shear: float = 2.0
    perspective: float = 0.0
    flipud: float = 0.3
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.15
    zoom_in_factor: float = 1.5
    zoom_out_factor: float = 0.5

    close_mosaic: int = 10

    workers: int = 8
    patience: int = 0
    save_period: int = -1
    resume: bool = False

    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5


@dataclass
class EvaluationConfig:
    """Evaluation settings for high recall focus."""
    # IoU thresholds for evaluation
    iou_thres: float = 0.3  # Lower IoU for higher recall (small objects)
    conf_thres: float = 0.001  # Very low confidence for high recall
    max_det: int = 300  # Max detections per image

    # Metrics to focus
    eval_metrics: List[str] = field(default_factory=lambda: [
        'precision', 'recall', 'mAP50', 'mAP50-95'
    ])


@dataclass
class ProjectConfig:
    """Project/output configuration."""
    project_name: str = 'TBM_Detection'
    experiment_name: str = 'ft_yolo11s_640_v2'
    save_dir: Path = Path(__file__).parent / "runs"
    exist_ok: bool = True  # Overwrite existing

    def __post_init__(self):
        # Convert to forward slashes for Windows compatibility
        save_str = str(self.save_dir).replace('\\', '/')
        self.save_dir = Path(save_str)
        self.experiment_name = str(self.experiment_name).replace('\\', '/')


def get_config() -> tuple:
    """Get all configurations."""
    model = ModelConfig()
    dataset = DatasetConfig()
    training = TrainingConfig()
    evaluation = EvaluationConfig()
    project = ProjectConfig()

    return model, dataset, training, evaluation, project


def print_config(model: ModelConfig, dataset: DatasetConfig,
                 training: TrainingConfig, evaluation: EvaluationConfig):
    """Print configuration summary."""
    print("=" * 60)
    print("TBM DETECTION - TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"\n[MODEL]")
    print(f"  Architecture: {model.model_name}")

    print(f"\n[DATASET]")
    print(f"  Path: {dataset.dataset_path}")
    print(f"  Image size: {dataset.img_size}")
    print(f"  Classes: {dataset.class_names}")

    print(f"\n[TRAINING]")
    print(f"  Epochs: {training.epochs}")
    print(f"  Batch size: {training.batch}")
    print(f"  Optimizer: {training.optimizer}")
    print(f"  Initial LR: {training.lr0}")
    print(f"  Mosaic: {training.mosaic} | MixUp: {training.mixup} | CopyPaste: {training.copy_paste}")

    print(f"\n[EVALUATION]")
    print(f"  IoU threshold: {evaluation.iou_thres}")
    print(f"  Conf threshold: {evaluation.conf_thres}")
    print("=" * 60)
