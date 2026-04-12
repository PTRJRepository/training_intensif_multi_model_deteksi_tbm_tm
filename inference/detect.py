import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def run_inference(
    weights=r"Train\runs\exp1_yolo11n_640\weights\best.pt",
    source=r"Dataset\dataset_pakai_ini\images\val",
    imgsz=640,
    conf=0.25,
    iou=0.7,
    device="",
    save=True,
    project="inference/runs",
    name="detect",
    exist_ok=True
):
    """
    Run inference using a trained YOLOv11 model.
    """
    # Load model
    model = YOLO(weights)
    
    # Run prediction
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=save,
        project=project,
        name=name,
        exist_ok=exist_ok
    )
    
    print(f"\nInference completed. Results saved to {os.path.abspath(os.path.join(project, name))}")
    
    # Return results for further processing if needed
    return results

def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Inference Script for Palm Tree Detection")
    parser.add_argument("--weights", type=str, default=r"Train\runs\exp1_yolo11n_640\weights\best.pt", help="Path to weights file")
    parser.add_argument("--source", type=str, default=r"Dataset\dataset_pakai_ini\images\val", help="Path to source (image, folder, video)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold for NMS")
    parser.add_argument("--device", type=str, default="", help="Device to run on (e.g. 0, 1 or cpu)")
    parser.add_argument("--save", action="store_true", default=True, help="Save results")
    parser.add_argument("--project", type=str, default="inference/runs", help="Project directory for results")
    parser.add_argument("--name", type=str, default="detect", help="Experiment name")
    
    args = parser.parse_args()
    
    run_inference(
        weights=args.weights,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        project=args.project,
        name=args.name
    )

if __name__ == "__main__":
    main()
