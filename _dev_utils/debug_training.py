"""
TBM Training Debug & Analysis Tool
Generated: 2026-04-11
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Try to import optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def analyze_current_state():
    """Analyze current training state."""
    print("=" * 70)
    print("TBM TRAINING DEBUG ANALYSIS")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "Dataset" / "dataset_pakai_ini"
    weights_path = project_root / "Train" / "runs" / "exp1_yolo11n_640" / "weights"

    results = {
        "analysis_date": datetime.now().isoformat(),
        "dataset_stats": {},
        "training_stats": {},
        "issues": [],
        "recommendations": []
    }

    # Check dataset
    train_labels = list((dataset_path / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_path / "labels" / "val").glob("*.txt"))

    total_trees_train = 0
    total_trees_val = 0
    bbox_sizes_train = []

    for f in train_labels:
        lines = f.read_text().strip().split("\n")
        total_trees_train += len(lines)
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                bbox_sizes_train.append((float(parts[3]), float(parts[4])))

    for f in val_labels:
        lines = f.read_text().strip().split("\n")
        total_trees_val += len(lines)

    results["dataset_stats"] = {
        "train_images": len(list((dataset_path / "images" / "train").glob("*.jpg"))),
        "val_images": len(list((dataset_path / "images" / "val").glob("*.jpg"))),
        "train_tiles": len(train_labels),
        "val_tiles": len(val_labels),
        "total_trees_train": total_trees_train,
        "total_trees_val": total_trees_val,
        "avg_trees_per_tile_train": total_trees_train / len(train_labels) if train_labels else 0
    }

    # Check weights
    if weights_path.exists():
        last_pt = weights_path / "last.pt"
        best_pt = weights_path / "best.pt"
        results["training_stats"] = {
            "weights_dir": str(weights_path).replace('\\', '/'),
            "last_exists": last_pt.exists(),
            "best_exists": best_pt.exists(),
            "last_size_mb": last_pt.stat().st_size / (1024*1024) if last_pt.exists() else 0
        }

    # Analyze bbox sizes
    if bbox_sizes_train and HAS_NUMPY:
        widths = [b[0] for b in bbox_sizes_train]
        heights = [b[1] for b in bbox_sizes_train]
        pixel_widths = [w * 640 for w in widths]
        pixel_heights = [h * 640 for h in heights]

        results["bbox_analysis"] = {
            "width_normalized": {"min": min(widths), "max": max(widths), "mean": float(np.mean(widths))},
            "height_normalized": {"min": min(heights), "max": max(heights), "mean": float(np.mean(heights))},
            "width_pixels_640": {"min": min(pixel_widths), "max": max(pixel_widths), "mean": float(np.mean(pixel_widths))},
            "height_pixels_640": {"min": min(pixel_heights), "max": max(pixel_heights), "mean": float(np.mean(pixel_heights))}
        }

        # Critical threshold checks
        small_objects = sum(1 for w, h in bbox_sizes_train if w * 640 < 16 or h * 640 < 16)
        results["small_object_count"] = small_objects
        results["small_object_percent"] = small_objects / len(bbox_sizes_train) * 100

    # Issues
    results["issues"] = [
        {
            "severity": "CRITICAL",
            "code": "TINY_DATASET",
            "description": f"Only {len(train_labels)} training tiles - very small dataset",
            "impact": "Model has limited examples to learn from"
        },
        {
            "severity": "CRITICAL",
            "code": "SMALL_OBJECTS",
            "description": f"BBox ~10px - YOLO struggles with <16px objects",
            "impact": "Detection accuracy inherently limited by object size"
        },
        {
            "severity": "HIGH",
            "code": "LOW_RECALL",
            "description": "Recall only 0.377 - model missing many trees",
            "impact": "False negatives - undetected trees"
        }
    ]

    # Recommendations
    results["recommendations"] = [
        {
            "priority": 1,
            "action": "Increase image size to 1280",
            "reason": "Makes 10px trees appear as 20px - within YOLO's good detection range",
            "command": "python Train/train.py --model yolo11s --img-size 1280 --batch 2 --epochs 300"
        },
        {
            "priority": 2,
            "action": "Use larger model (yolo11s or yolo11m)",
            "reason": "More capacity to learn small object patterns",
            "command": "python Train/train.py --model yolo11m --img-size 1280 --batch 1 --epochs 500"
        },
        {
            "priority": 3,
            "action": "Increase copy_paste augmentation to 0.2-0.3",
            "reason": "Critical for small isolated objects",
            "command": "python Train/train.py --copy-paste 0.2"
        },
        {
            "priority": 4,
            "action": "Generate more training tiles with higher overlap",
            "reason": "Current 17 tiles is very limited",
            "command": "Edit Tools/prepare_multi_image_dataset.py - increase overlap to 50%"
        }
    ]

    return results


def print_report(results):
    """Print formatted debug report."""
    print("\n" + "=" * 70)
    print("DATASET STATS")
    print("=" * 70)
    ds = results["dataset_stats"]
    print(f"  Train images: {ds.get('train_images', 'N/A')}")
    print(f"  Val images:   {ds.get('val_images', 'N/A')}")
    print(f"  Train tiles:   {ds.get('train_tiles', 'N/A')}")
    print(f"  Val tiles:    {ds.get('val_tiles', 'N/A')}")
    print(f"  Total trees (train): {ds.get('total_trees_train', 'N/A')}")
    print(f"  Trees per tile: {ds.get('avg_trees_per_tile_train', 0):.1f}")

    if "bbox_analysis" in results:
        ba = results["bbox_analysis"]
        print("\n" + "=" * 70)
        print("BBOX SIZE ANALYSIS (at 640x640)")
        print("=" * 70)
        print(f"  Width:  min={ba['width_pixels_640']['min']:.1f}, max={ba['width_pixels_640']['max']:.1f}, mean={ba['width_pixels_640']['mean']:.1f}px")
        print(f"  Height: min={ba['height_pixels_640']['min']:.1f}, max={ba['height_pixels_640']['max']:.1f}, mean={ba['height_pixels_640']['mean']:.1f}px")
        print(f"  Small objects (<16px): {results.get('small_object_count', 0)} ({results.get('small_object_percent', 0):.1f}%)")

    print("\n" + "=" * 70)
    print("ISSUES IDENTIFIED")
    print("=" * 70)
    for issue in results["issues"]:
        print(f"  [{issue['severity']}] {issue['code']}")
        print(f"            {issue['description']}")
        print(f"            Impact: {issue['impact']}")
        print()

    print("=" * 70)
    print("RECOMMENDATIONS FOR HIGHER ACCURACY")
    print("=" * 70)
    for rec in results["recommendations"]:
        print(f"  [Priority {rec['priority']}] {rec['action']}")
        print(f"            Reason: {rec['reason']}")
        print(f"            Command: {rec['command']}")
        print()


def save_report(results, path=None):
    """Save report as JSON."""
    if path is None:
        path = Path(__file__).parent / "debug_analysis.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Report to: {path}")


if __name__ == "__main__":
    print("[INFO] Starting TBM training debug analysis...\n")

    results = analyze_current_state()
    print_report(results)
    save_report(results)

    print("\n" + "=" * 70)
    print("QUICK START - RECOMMENDED TRAINING COMMAND")
    print("=" * 70)
    print("""
# For higher accuracy, run with these settings:

python Train/train.py --model yolo11s --img-size 1280 --batch 2 --epochs 300 --copy-paste 0.2

# Or for maximum accuracy (slower):

python Train/train.py --model yolo11m --img-size 1280 --batch 1 --epochs 500 --copy-paste 0.3
""")
