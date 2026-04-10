"""
Evaluation script for TBM Detection Model.
Focus: High recall analysis and detection quality metrics.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Train.config import get_config, EvaluationConfig, DatasetConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TBM Detection Model')

    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights (.pt file)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')

    # Detection thresholds (CRITICAL for recall tuning)
    parser.add_argument('--conf', type=float, default=0.001,
                       help='Confidence threshold (lower = higher recall)')
    parser.add_argument('--iou', type=float, default=0.3,
                       help='IoU threshold for NMS (lower = more detections)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--save-json', action='store_true',
                       help='Save results to JSON')
    parser.add_argument('--plot', action='store_true',
                       help='Plot confusion matrix and curves')

    return parser.parse_args()


def compute_recall_precision(y_true, y_pred, iou_threshold=0.3):
    """
    Compute recall and precision manually.

    Args:
        y_true: List of ground truth boxes [[x1,y1,x2,y2], ...]
        y_pred: List of predicted boxes [[x1,y1,x2,y2,conf], ...]
        iou_threshold: IoU threshold for match
    """
    if len(y_true) == 0:
        return 0.0, 0.0, len(y_pred)

    if len(y_pred) == 0:
        return 0.0, 0.0, 0

    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(y_true, y_pred)

    # Match predictions to ground truth
    matched_gt = set()
    matched_pred = set()
    matches = []

    # Sort by confidence
    conf_order = np.argsort([-p[4] for p in y_pred])

    for pred_idx in conf_order:
        pred_box = y_pred[pred_idx]
        best_iou = iou_threshold
        best_gt_idx = -1

        for gt_idx in range(len(y_true)):
            if gt_idx in matched_gt:
                continue

            iou = iou_matrix[gt_idx, pred_idx]
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            matches.append((best_gt_idx, pred_idx, best_iou))

    # Calculate metrics
    true_positives = len(matches)
    false_positives = len(y_pred) - true_positives
    false_negatives = len(y_true) - true_positives

    recall = true_positives / len(y_true) if len(y_true) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    return recall, precision, true_positives, false_positives, false_negatives


def compute_iou_matrix(boxes1, boxes2):
    """Compute IoU between two sets of boxes."""
    n1 = len(boxes1)
    n2 = len(boxes2)
    iou_matrix = np.zeros((n1, n2))

    for i, b1 in enumerate(boxes1):
        x1_1, y1_1, x2_1, y2_1 = b1[:4]
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)

        for j, b2 in enumerate(boxes2):
            x1_2, y1_2, x2_2, y2_2 = b2[:4]
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

            # Intersection
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union_area = area1 + area2 - inter_area

            iou_matrix[i, j] = inter_area / union_area if union_area > 0 else 0

    return iou_matrix


def evaluate_confidence_thresholds(model, data_yaml, img_size=640, conf_range=None):
    """
    Evaluate model at multiple confidence thresholds.
    Critical for understanding recall vs precision trade-off.
    """
    if conf_range is None:
        conf_range = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

    print("\n" + "=" * 70)
    print("CONFIDENCE THRESHOLD ANALYSIS (Recall Focus)")
    print("=" * 70)
    print(f"{'Conf':<10} {'Recall':<12} {'Precision':<12} {'Detections':<12}")
    print("-" * 70)

    results = []
    for conf in conf_range:
        # Run validation with specific conf threshold
        metrics = model.val(
            data=data_yaml,
            imgsz=img_size,
            conf=conf,
            verbose=False
        )

        recall = getattr(metrics, 'r', 0)  # Recall
        precision = getattr(metrics, 'p', 0)  # Precision
        detections = getattr(metrics, 'box', {}).get('tp', 0) + getattr(metrics, 'box', {}).get('fp', 0)

        results.append({
            'conf': conf,
            'recall': recall,
            'precision': precision,
            'detections': detections
        })

        print(f"{conf:<10.3f} {recall:<12.4f} {precision:<12.4f} {detections:<12}")

    # Find threshold with best recall (while maintaining reasonable precision)
    best_for_recall = max(results, key=lambda x: x['recall'])
    best_f1 = max(results, key=lambda x: 2 * x['recall'] * x['precision'] / (x['recall'] + x['precision'] + 1e-6))

    print("-" * 70)
    print(f"\n[BEST RECALL] Conf={best_for_recall['conf']:.3f}, Recall={best_for_recall['recall']:.4f}")
    print(f"[BEST F1]     Conf={best_f1['conf']:.3f}, F1={2*best_f1['recall']*best_f1['precision']/(best_f1['recall']+best_f1['precision']+1e-6):.4f}")

    return results


def main():
    args = parse_args()

    # Validate weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        sys.exit(1)

    # Get dataset config
    _, dataset_cfg, _, evaluation_cfg, _ = get_config()
    data_yaml = args.data or str(dataset_cfg.data_yaml)

    if not Path(data_yaml).exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        sys.exit(1)

    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)

    # Load model
    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    # Standard evaluation
    print("\n" + "=" * 70)
    print("STANDARD EVALUATION")
    print("=" * 70)

    metrics = model.val(
        data=data_yaml,
        split=args.split,
        imgsz=args.img_size,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
        plots=args.plot
    )

    print(f"\n[METRICS]")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Recall:   {metrics.box.r:.4f}")
    print(f"  Precision: {metrics.box.p:.4f}")

    # Confidence threshold analysis for recall tuning
    print("\n[INFO] Running confidence threshold analysis...")
    threshold_results = evaluate_confidence_thresholds(
        model, data_yaml, args.img_size
    )

    # Save results if requested
    if args.save_json:
        import json
        output_path = Path(args.output) if args.output else weights_path.parent / "eval_results.json"

        all_results = {
            'weights': str(weights_path),
            'standard_metrics': {
                'map50': float(metrics.box.map50),
                'map': float(metrics.box.map),
                'recall': float(metrics.box.r),
                'precision': float(metrics.box.p)
            },
            'threshold_analysis': threshold_results
        }

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n[INFO] Results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("[COMPLETE] Evaluation finished!")
    print("=" * 70)


if __name__ == '__main__':
    main()
