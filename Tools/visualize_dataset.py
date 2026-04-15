import cv2
import os
import yaml
import argparse
import random
from tqdm import tqdm
from pathlib import Path

"""
dataset_visualizer.py
--------------------
A tool to visualize YOLO-formatted datasets.
Draws bounding boxes on images and saves them to an output folder.
"""

def get_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO Dataset")
    parser.add_argument("--data", type=str, default="Dataset/dataset_pakai_ini/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to visualize")
    parser.add_argument("--num", type=int, default=5, help="Number of images to visualize")
    parser.add_argument("--output", type=str, default="visualized_output", help="Output directory")
    parser.add_argument("--thickness", type=int, default=1, help="Bounding box thickness")
    parser.add_argument("--no_label", action="store_true", help="Don't draw class labels")
    return parser.parse_args()

def draw_boxes(image, labels, names, thickness, draw_label=True):
    h, w, _ = image.shape

    for label in labels:
        parts = label.strip().split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(parts[0])
            xc, yc, nw, nh = map(float, parts[1:5])

            # Convert YOLO to absolute coordinates (center point only)
            cx = int(xc * w)
            cy = int(yc * h)

            # Draw small red dot
            cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)
        except Exception as e:
            print(f"Error parsing label: {label}. Error: {e}")
            continue

    return image

def main():
    # Set relative working directory to script's parent if needed
    # But here we assume it's run from the project root.
    
    args = get_args()
    
    # Resolve data.yaml path
    data_path = Path(args.data)
    if not data_path.exists():
        # Try relative to the script location if not found from CWD
        script_dir = Path(__file__).parent.parent
        data_path = script_dir / args.data
        
    if not data_path.exists():
        print(f"Error: {args.data} not found. Please provide correct path with --data")
        return

    print(f"Loading config from: {data_path}")
    try:
        with open(data_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading YAML: {e}")
        return
        
    names = data_cfg.get('names', {})
    dataset_root = data_path.parent
    
    split_path_rel = data_cfg.get(args.split)
    if not split_path_rel:
        print(f"Split '{args.split}' path not found in data.yaml")
        return
        
    # YOLO paths in data.yaml are often relative to the 'path' entry or dataset root
    cfg_base_path = data_cfg.get('path')
    if cfg_base_path:
        # data.yaml might have a 'path' entry
        img_dir = Path(cfg_base_path) / split_path_rel
    else:
        img_dir = dataset_root / split_path_rel
        
    # If path from yaml doesn't exist, try relative to data_path
    if not img_dir.exists():
        img_dir = dataset_root / split_path_rel
    
    # Determine label directory (usually maps images/ -> labels/)
    # We check if 'images' is in the path and replace it with 'labels'
    label_dir_str = str(img_dir).replace("images", "labels")
    label_dir = Path(label_dir_str)
    
    if not img_dir.exists():
        print(f"Image directory not found: {img_dir}")
        return
        
    print(f"Scanning images in: {img_dir}")
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts]
    
    if not img_files:
        print(f"No images found in {img_dir}")
        return
        
    print(f"Found {len(img_files)} total images. Sampling {min(args.num, len(img_files))}...")
    
    random.seed(42) # For reproducible sampling
    num_to_viz = min(args.num, len(img_files))
    selected_files = random.sample(img_files, num_to_viz)
    
    output_dir = Path(args.output) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(selected_files)
    count_with_labels = 0
    
    for img_path in pbar:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        pbar.set_description(f"Processing {img_path.name}")
        
        # Look for label file
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = f.readlines()
            if labels:
                img = draw_boxes(img, labels, names, args.thickness, not args.no_label)
                count_with_labels += 1
        
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        
    print(f"\nSummary:")
    print(f"- Total processed: {len(selected_files)}")
    print(f"- Labeled images:  {count_with_labels}")
    print(f"- Output folder:   {output_dir.absolute()}")
    print(f"\nUse common image viewer to inspect the results.")

if __name__ == "__main__":
    main()
