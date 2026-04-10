"""
Inference script for TBM Detection.
Multi-model inference support for ensemble predictions.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Union
import time

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='TBM Detection Inference')

    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights (.pt file or comma-separated for ensemble)')
    parser.add_argument('--source', type=str, required=True,
                       help='Image or directory to process')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.3,
                       help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save results to txt (YOLO format)')
    parser.add_argument('--save-conf', action='store_true',
                       help='Save confidence scores in txt')
    parser.add_argument('--augment', action='store_true',
                       help='Use TTA (Test Time Augmentation)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device (0 for GPU)')

    return parser.parse_args()


class TBMDetector:
    """TBM Detector with multi-model support."""

    def __init__(self, weights: Union[str, List[str]], device='0'):
        """Initialize detector with one or more models."""
        from ultralytics import YOLO

        self.models = []
        self.device = device

        # Single model
        if isinstance(weights, str):
            weights = [weights]

        # Load multiple models
        for w in weights:
            print(f"[INFO] Loading model: {w}")
            model = YOLO(w)
            model.to(device)
            self.models.append(model)

        print(f"[INFO] Loaded {len(self.models)} model(s)")

    def predict(self, image, conf=0.1, iou=0.3, augment=False):
        """
        Run inference on single image.
        Returns combined predictions from all models.
        """
        all_boxes = []
        all_confs = []
        all_classes = []

        for model in self.models:
            results = model.predict(
                image,
                conf=conf,
                iou=iou,
                augment=augment,
                verbose=False
            )

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()

                    all_boxes.append(boxes)
                    all_confs.append(confs)
                    all_classes.append(classes)

        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Concatenate all predictions
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_confs = np.concatenate(all_confs, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)

        # Apply weighted averaging for overlapping boxes (ensemble)
        if len(self.models) > 1:
            all_boxes, all_confs, all_classes = self._nms_ensemble(
                all_boxes, all_confs, all_classes, iou
            )

        return all_boxes, all_confs, all_classes

    def _nms_ensemble(self, boxes, confs, classes, iou_thres):
        """Non-maximum suppression for ensemble."""
        if len(boxes) == 0:
            return boxes, confs, classes

        # Sort by confidence
        order = np.argsort(-confs)

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            iou = self._box_iou(boxes[i], boxes[order[1:]])

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        return boxes[keep], confs[keep], classes[keep]

    def _box_iou(self, box, boxes):
        """Compute IoU between one box and multiple boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = area1 + area2 - inter
        iou = inter / (union + 1e-6)

        return iou

    def count_trees(self, image, conf=0.1, iou=0.3):
        """Count trees in image."""
        boxes, confs, classes = self.predict(image, conf, iou)
        return len(boxes), boxes, confs


def process_image(detector, image_path, output_dir, args):
    """Process single image and save results."""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARNING] Could not read: {image_path}")
        return 0

    # Run inference
    start_time = time.time()
    tree_count, boxes, confs = detector.count_trees(
        image, conf=args.conf, iou=args.iou
    )
    inference_time = time.time() - start_time

    # Draw boxes
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{conf:.2f}', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Add count text
    cv2.putText(image, f'Trees: {tree_count}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save result
    if output_dir:
        output_path = output_dir / f"{image_path.stem}_result.jpg"
        cv2.imwrite(str(output_path), image)

        # Save txt
        if args.save_txt:
            txt_path = output_dir / f"{image_path.stem}.txt"
            with open(txt_path, 'w') as f:
                for box, conf, cls in zip(boxes, confs, [0]*len(boxes)):
                    x1, y1, x2, y2 = box
                    x_center = ((x1 + x2) / 2) / image.shape[1]
                    y_center = ((y1 + y2) / 2) / image.shape[0]
                    w = (x2 - x1) / image.shape[1]
                    h = (y2 - y1) / image.shape[0]
                    if args.save_conf:
                        f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
                    else:
                        f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"  {image_path.name}: {tree_count} trees ({inference_time:.3f}s)")

    return tree_count


def process_directory(detector, source_dir, output_dir, args):
    """Process all images in directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(source_dir).glob(f"*{ext}"))
        image_files.extend(Path(source_dir).glob(f"*{ext.upper()}"))

    if len(image_files) == 0:
        print(f"[WARNING] No images found in {source_dir}")
        return 0

    print(f"[INFO] Found {len(image_files)} images")

    total_trees = 0
    for image_path in image_files:
        count = process_image(detector, image_path, output_dir, args)
        total_trees += count

    return total_trees


def main():
    args = parse_args()

    # Parse weights (single or comma-separated for ensemble)
    weights_list = [w.strip() for w in args.weights.split(',')]

    # Validate weights
    for w in weights_list:
        if not Path(w).exists():
            print(f"[ERROR] Weights not found: {w}")
            sys.exit(1)

    # Create output directory
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    print(f"[INFO] Initializing TBM Detector with {len(weights_list)} model(s)...")
    detector = TBMDetector(weights_list, device=args.device)

    # Process source
    source_path = Path(args.source)

    if source_path.is_file():
        # Single file
        image = cv2.imread(str(source_path))
        count, _, _ = detector.count_trees(image, conf=args.conf, iou=args.iou)
        process_image(detector, source_path, output_dir, args)
        print(f"\n[SUMMARY] {source_path.name}: {count} trees detected")

    elif source_path.is_dir():
        # Directory
        total = process_directory(detector, source_path, output_dir, args)
        print(f"\n[SUMMARY] Total: {total} trees detected in {source_path}")

    else:
        print(f"[ERROR] Source not found: {source_path}")
        sys.exit(1)

    print("\n[COMPLETE] Inference finished!")


if __name__ == '__main__':
    main()
