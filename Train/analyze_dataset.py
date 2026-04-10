"""
Dataset analysis utility for TBM Detection.
Analyzes tree distribution, density, and provides insights for training.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import rasterio
import geopandas as gpd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze TBM Dataset')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to dataset (default: Dataset/dataset_pakai_ini)')
    return parser.parse_args()


def analyze_shapefile(shapefile_path):
    """Analyze shapefile annotations."""
    print(f"\n[INFO] Analyzing shapefile: {shapefile_path}")

    gdf = gpd.read_file(shapefile_path)

    print(f"\n[SHP ANALYSIS]")
    print(f"  Total features: {len(gdf)}")
    print(f"  CRS: {gdf.crs}")
    print(f"  Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")

    # Count valid geometries
    valid = gdf.geometry.notna().sum()
    print(f"  Valid geometries: {valid}/{len(gdf)}")

    return gdf


def analyze_image(image_path):
    """Analyze TIFF image properties."""
    print(f"\n[INFO] Analyzing image: {image_path}")

    with rasterio.open(image_path) as src:
        print(f"\n[IMAGE ANALYSIS]")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  Bands: {src.count}")
        print(f"  CRS: {src.crs}")
        print(f"  Resolution: {src.res}")
        print(f"  Data type: {src.dtypes[0]}")

        # Compute statistics
        image = src.read()
        for i in range(min(image.shape[0], 4)):
            band = image[i].astype(np.float32)
            valid = band[band != 0]
            if len(valid) > 0:
                print(f"  Band {i+1}: min={valid.min():.1f}, max={valid.max():.1f}, mean={valid.mean():.1f}")

    return src


def analyze_tile_labels(dataset_path, split='train'):
    """Analyze YOLO label files in a split."""
    label_dir = Path(dataset_path) / 'labels' / split

    if not label_dir.exists():
        print(f"[WARNING] Label directory not found: {label_dir}")
        return None

    label_files = list(label_dir.glob('*.txt'))

    if len(label_files) == 0:
        print(f"[WARNING] No label files found in {label_dir}")
        return None

    print(f"\n[TILE ANALYSIS - {split.upper()}]")
    print(f"  Total tiles: {len(label_files)}")

    # Count trees per tile
    tree_counts = []
    bbox_sizes = []

    for label_file in tqdm(label_files, desc="Analyzing labels"):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        tree_counts.append(len(lines))

        # Analyze bbox sizes
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                w, h = float(parts[3]), float(parts[4])
                bbox_sizes.append((w, h))

    print(f"  Total trees: {sum(tree_counts)}")
    print(f"  Trees per tile: min={min(tree_counts)}, max={max(tree_counts)}, mean={np.mean(tree_counts):.1f}")

    if bbox_sizes:
        widths = [b[0] for b in bbox_sizes]
        heights = [b[1] for b in bbox_sizes]
        print(f"  BBox width (normalized): min={min(widths):.4f}, max={max(widths):.4f}, mean={np.mean(widths):.4f}")
        print(f"  BBox height (normalized): min={min(heights):.4f}, max={max(heights):.4f}, mean={np.mean(heights):.4f}")

        # Estimate pixel sizes at 640x640
        pixel_widths = [w * 640 for w in widths]
        pixel_heights = [h * 640 for h in heights]
        print(f"  BBox width (pixels @640): min={min(pixel_widths):.1f}, max={max(pixel_widths):.1f}, mean={np.mean(pixel_widths):.1f}")
        print(f"  BBox height (pixels @640): min={min(pixel_heights):.1f}, max={max(pixel_heights):.1f}, mean={np.mean(pixel_heights):.1f}")

    return {
        'tile_count': len(label_files),
        'total_trees': sum(tree_counts),
        'tree_counts': tree_counts,
        'bbox_sizes': bbox_sizes
    }


def analyze_tree_density(dataset_path):
    """Analyze spatial density of trees."""
    label_dir = Path(dataset_path) / 'labels' / 'train'
    image_dir = Path(dataset_path) / 'images' / 'train'

    if not label_dir.exists():
        return None

    label_files = list(label_dir.glob('*.txt'))

    densities = []
    nearest_distances = []

    for label_file in tqdm(label_files, desc="Analyzing density"):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            continue

        # Get tree positions
        points = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y = float(parts[1]), float(parts[2])
                points.append((x, y))

        if len(points) < 2:
            continue

        # Compute density
        area = 1.0  # Normalized
        density = len(points) / area
        densities.append(density)

        # Find nearest neighbor distances
        for i, p1 in enumerate(points):
            min_dist = float('inf')
            for j, p2 in enumerate(points):
                if i != j:
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    min_dist = min(min_dist, dist)
            nearest_distances.append(min_dist)

    print(f"\n[DENSITY ANALYSIS]")
    print(f"  Trees per tile (normalized area): min={min(densities):.0f}, max={max(densities):.0f}, mean={np.mean(densities):.0f}")
    print(f"  Nearest neighbor distance: min={min(nearest_distances):.4f}, max={max(nearest_distances):.4f}, mean={np.mean(nearest_distances):.4f}")

    # Convert to approximate pixels
    avg_nn_pixel = np.mean(nearest_distances) * 640
    print(f"  Avg nearest neighbor @640px: {avg_nn_pixel:.1f} pixels")

    return {
        'densities': densities,
        'nearest_distances': nearest_distances,
        'avg_nn_pixel': avg_nn_pixel
    }


def analyze_augmentation_needs(dataset_path):
    """Suggest augmentation strategy based on analysis."""
    print("\n" + "=" * 70)
    print("AUGMENTATION RECOMMENDATIONS")
    print("=" * 70)

    print("""
Based on TBM detection challenges:

[CRITICAL FOR SMALL OBJECTS]
1. MOSAIC AUGMENTATION - ENABLE (already default)
   - Helps model learn context at tile edges
   - Combines multiple tiles into one

2. COPY-PASTE AUGMENTATION - RECOMMEND 0.1-0.2
   - Duplicates small objects to increase visibility
   - Critical for trees that appear rarely in tiles

3. MIXUP - USE MODERATELY (0.05-0.1)
   - Helps with overlapping tree patterns

[FOCUS ON RECALL]
- Use LOW confidence threshold during inference (0.001-0.01)
- Use LOWER IoU threshold for NMS (0.3-0.4)
- This catches more trees but may increase false positives

[ANCHOR SETTINGS]
- Default YOLO anchors work well for ~10px objects
- If recall is low, try smaller anchor boxes

[HARD NEGATIVE MINING]
- Consider adding images WITHOUT trees as negative samples
- Helps reduce false positives on empty areas
""")

    return {
        'mosaic': 1.0,
        'copy_paste': 0.15,
        'mixup': 0.1,
        'conf_thres': 0.001,
        'iou_thres': 0.3
    }


def main():
    args = parse_args()

    # Default dataset path
    if args.dataset is None:
        dataset_path = Path(__file__).parent.parent / "Dataset" / "dataset_pakai_ini"
    else:
        dataset_path = Path(args.dataset)

    print("=" * 70)
    print("TBM DATASET ANALYSIS")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")

    if not dataset_path.exists():
        print(f"\n[ERROR] Dataset not found: {dataset_path}")
        print("\nPlease run preparation first:")
        print("  python Tools/prepare_multi_image_dataset.py")
        sys.exit(1)

    # Analyze train and val splits
    for split in ['train', 'val']:
        results = analyze_tile_labels(dataset_path, split)

    # Analyze density
    density_results = analyze_tree_density(dataset_path)

    # Get source data info
    print("\n" + "=" * 70)
    print("SOURCE DATA")
    print("=" * 70)

    # Try to analyze source images
    raw_dataset = Path(__file__).parent.parent / "Dataset" / "Raw Dataset"
    clipped_dir = raw_dataset / "TBM" / "Processed" / "Clipped"

    if clipped_dir.exists():
        tiff_files = list(clipped_dir.glob("*.tif"))
        for tiff in tiff_files:
            try:
                analyze_image(tiff)
            except Exception as e:
                print(f"[WARNING] Could not analyze {tiff}: {e}")

    # Analyze shapefiles
    shapefile_dir = raw_dataset / "TBM"
    for shp in shapefile_dir.glob("*.shp"):
        if shp.name not in ['Potong_layer_1A.shp']:
            try:
                analyze_shapefile(shp)
            except Exception as e:
                print(f"[WARNING] Could not analyze {shp}: {e}")

    # Recommendations
    recommendations = analyze_augmentation_needs(dataset_path)

    print("\n" + "=" * 70)
    print("[COMPLETE] Dataset analysis finished!")
    print("=" * 70)

    return recommendations


if __name__ == '__main__':
    main()
