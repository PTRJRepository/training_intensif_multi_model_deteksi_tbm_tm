"""
Dataset preparation for small object detection (palm tree counting from drone imagery).
Supports multiple strategies:
1. Full frame training (recommended)
2. Tiled/chip training (for very large images)
3. Augmented training (zoom-in crops for small objects)
"""

import os
import cv2
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping, box
from pathlib import Path
import shutil
from tqdm import tqdm
import random


class PalmTreeDatasetPreparer:
    """Prepare dataset for palm tree detection from drone imagery and shapefiles."""
    
    def __init__(self, image_path, shapefile_path, output_dir, image_size=640):
        """
        Initialize the dataset preparer.
        
        Args:
            image_path: Path to the drone TIFF image
            shapefile_path: Path to shapefile with tree locations (points)
            output_dir: Directory to save the prepared dataset
            image_size: Target image size for training (default: 640 for YOLO)
        """
        self.image_path = image_path
        self.shapefile_path = shapefile_path
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        
        # Create output directories
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
    def read_image_and_annotations(self):
        """Read the TIFF image and shapefile annotations."""
        print("📖 Reading TIFF image...")
        with rasterio.open(self.image_path) as src:
            self.image = src.read()  # Shape: (bands, height, width)
            self.image_meta = src.meta
            self.transform = src.transform
            self.crs = src.crs
            print(f"   Image shape: {self.image.shape}")
            print(f"   Image dtype: {self.image.dtype}")
        
        # Transpose to (height, width, bands) for easier processing
        if self.image.shape[0] <= 4:  # Assuming <= 4 bands
            self.image = np.transpose(self.image, (1, 2, 0))
        
        print("📖 Reading shapefile annotations...")
        self.gdf = gpd.read_file(self.shapefile_path)
        print(f"   Found {len(self.gdf)} tree annotations")
        
        # Reproject if needed
        if self.gdf.crs != self.crs:
            print(f"   Reprojecting from {self.gdf.crs} to {self.crs}")
            self.gdf = self.gdf.to_crs(self.crs)
        
        # Convert geometry to pixel coordinates
        self.tree_points = self._geom_to_pixel_coords()
        print(f"   Converted {len(self.tree_points)} points to pixel coordinates")
        
    def _geom_to_pixel_coords(self):
        """Convert geographic coordinates to pixel coordinates."""
        points = []
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Point':
                px, py = ~self.transform * (geom.x, geom.y)
                points.append({
                    'x': int(px),
                    'y': int(py),
                    'original_idx': idx
                })
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                # For polygons, use centroid
                centroid = geom.centroid
                px, py = ~self.transform * (centroid.x, centroid.y)
                points.append({
                    'x': int(px),
                    'y': int(py),
                    'original_idx': idx
                })
        return points
    
    def create_tiled_dataset(self, tile_size=640, overlap=0.2, val_split=0.2):
        """
        Create a tiled dataset by splitting large image into smaller chips.
        BEST FOR: Very large drone images where trees are small
        
        Args:
            tile_size: Size of each tile (default: 640)
            overlap: Overlap between tiles (default: 0.2 = 20%)
            val_split: Fraction for validation (default: 0.2)
        """
        print("\n🔲 Creating tiled dataset...")
        print(f"   Tile size: {tile_size}x{tile_size}")
        print(f"   Overlap: {overlap*100}%")
        
        step = int(tile_size * (1 - overlap))
        img_height, img_width = self.image.shape[:2]
        
        tiles = []
        tile_id = 0
        
        # Generate tiles
        for y in range(0, img_height - tile_size + 1, step):
            for x in range(0, img_width - tile_size + 1, step):
                # Extract tile
                tile_img = self.image[y:y+tile_size, x:x+tile_size]
                
                # Find trees in this tile
                trees_in_tile = []
                for tree in self.tree_points:
                    tx, ty = tree['x'] - x, tree['y'] - y
                    if 0 <= tx < tile_size and 0 <= ty < tile_size:
                        # Convert to YOLO format: (class, x_center, y_center, width, height)
                        # Normalized to [0, 1]
                        x_center = tx / tile_size
                        y_center = ty / tile_size
                        
                        # For point annotations, use small bbox (e.g., 10x10 pixels)
                        bbox_size = 10  # Adjust based on your tree size
                        w = bbox_size / tile_size
                        h = bbox_size / tile_size
                        
                        trees_in_tile.append({
                            'class': 0,  # Single class: tree
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': w,
                            'height': h
                        })
                
                # Only keep tiles with at least one tree
                if len(trees_in_tile) > 0:
                    tiles.append({
                        'id': tile_id,
                        'image': tile_img,
                        'x': x,
                        'y': y,
                        'trees': trees_in_tile
                    })
                    tile_id += 1
        
        print(f"   Generated {len(tiles)} tiles with trees")
        
        # Split into train/val
        random.shuffle(tiles)
        val_count = max(1, int(len(tiles) * val_split))
        train_tiles = tiles[val_count:]
        val_tiles = tiles[:val_count]
        
        print(f"   Train tiles: {len(train_tiles)}")
        print(f"   Val tiles: {len(val_tiles)}")
        
        # Save tiles
        self._save_tiles(train_tiles, "train")
        self._save_tiles(val_tiles, "val")
        
        # Create YOLO config
        self._create_yolo_config()
        
        return tiles
    
    def _save_tiles(self, tiles, split):
        """Save tiles to disk in YOLO format."""
        print(f"💾 Saving {split} tiles...")
        
        for tile in tqdm(tiles, desc=f"Saving {split}"):
            tile_id = tile['id']
            
            # Save image
            img_filename = f"tile_{tile_id:05d}.jpg"
            img_path = self.output_dir / "images" / split / img_filename
            
            # Convert to BGR for saving (if RGB)
            if tile['image'].shape[2] == 3:
                cv2.imwrite(str(img_path), cv2.cvtColor(tile['image'], cv2.COLOR_RGB2BGR))
            else:
                # Handle different number of bands
                if tile['image'].shape[2] == 1:
                    cv2.imwrite(str(img_path), tile['image'][:, :, 0])
                else:
                    # Take first 3 bands
                    cv2.imwrite(str(img_path), cv2.cvtColor(tile['image'][:, :, :3], cv2.COLOR_RGB2BGR))
            
            # Save label (YOLO format)
            label_filename = f"tile_{tile_id:05d}.txt"
            label_path = self.output_dir / "labels" / split / label_filename
            
            with open(label_path, 'w') as f:
                for tree in tile['trees']:
                    # YOLO format: class x_center y_center width height
                    f.write(f"{tree['class']} {tree['x_center']:.6f} {tree['y_center']:.6f} "
                           f"{tree['width']:.6f} {tree['height']:.6f}\n")
    
    def create_augmented_dataset(self, base_size=640, zoom_levels=[1.0, 1.5, 2.0], 
                                 random_crops=5, val_split=0.2):
        """
        Create augmented dataset with multiple zoom levels.
        BEST FOR: Small objects - helps model learn at different scales
        
        Args:
            base_size: Base image size
            zoom_levels: Different zoom factors (1.0 = original, 2.0 = 2x closer)
            random_crops: Number of random crops per zoom level
            val_split: Fraction for validation
        """
        print("\n🔍 Creating augmented dataset with multi-scale training...")
        print(f"   Zoom levels: {zoom_levels}")
        print(f"   Random crops per zoom: {random_crops}")
        
        img_height, img_width = self.image.shape[:2]
        all_samples = []
        sample_id = 0
        
        # Multi-scale zoom approach
        for zoom in zoom_levels:
            print(f"\n   📸 Processing zoom level: {zoom}x")
            
            # Calculate crop size for this zoom level
            crop_size = int(base_size / zoom)
            crop_size = min(crop_size, min(img_height, img_width))
            
            # Generate random crops
            for crop_idx in range(random_crops):
                # Random position
                if img_height - crop_size <= 0 or img_width - crop_size <= 0:
                    continue
                    
                y = random.randint(0, img_height - crop_size)
                x = random.randint(0, img_width - crop_size)
                
                # Extract crop
                crop_img = self.image[y:y+crop_size, x:x+crop_size]
                
                # Find trees in this crop
                trees = []
                for tree in self.tree_points:
                    tx, ty = tree['x'] - x, tree['y'] - y
                    if 0 <= tx < crop_size and 0 <= ty < crop_size:
                        x_center = tx / crop_size
                        y_center = ty / crop_size
                        bbox_size = 10 / crop_size  # Normalized
                        
                        trees.append({
                            'class': 0,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': bbox_size,
                            'height': bbox_size
                        })
                
                if len(trees) > 0:
                    # Resize to base_size
                    resized = cv2.resize(crop_img, (base_size, base_size))
                    
                    all_samples.append({
                        'id': sample_id,
                        'image': resized,
                        'trees': trees,
                        'zoom': zoom,
                        'crop_idx': crop_idx
                    })
                    sample_id += 1
        
        print(f"\n   Generated {len(all_samples)} augmented samples")
        
        # Split train/val
        random.shuffle(all_samples)
        val_count = max(1, int(len(all_samples) * val_split))
        train_samples = all_samples[val_count:]
        val_samples = all_samples[:val_count]
        
        print(f"   Train samples: {len(train_samples)}")
        print(f"   Val samples: {len(val_samples)}")
        
        # Save samples
        self._save_samples(train_samples, "train")
        self._save_samples(val_samples, "val")
        
        # Create YOLO config
        self._create_yolo_config()
        
        return all_samples
    
    def _save_samples(self, samples, split):
        """Save samples to disk."""
        print(f"💾 Saving {split} samples...")
        
        for sample in tqdm(samples, desc=f"Saving {split}"):
            sample_id = sample['id']
            
            # Save image
            img_filename = f"sample_{sample_id:05d}.jpg"
            img_path = self.output_dir / "images" / split / img_filename
            
            if sample['image'].shape[2] == 3:
                cv2.imwrite(str(img_path), cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(img_path), sample['image'][:, :, 0])
            
            # Save label
            label_filename = f"sample_{sample_id:05d}.txt"
            label_path = self.output_dir / "labels" / split / label_filename
            
            with open(label_path, 'w') as f:
                for tree in sample['trees']:
                    f.write(f"{tree['class']} {tree['x_center']:.6f} {tree['y_center']:.6f} "
                           f"{tree['width']:.6f} {tree['height']:.6f}\n")
    
    def _create_yolo_config(self):
        """Create YOLO training configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'palm_tree'
            }
        }
        
        config_path = self.output_dir / "data.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n📝 Created YOLO config: {config_path}")
        print(f"   Dataset ready for training with YOLOv8/v11!")


def main():
    """Main function with multiple strategies."""
    
    # Paths
    IMAGE_PATH = r"D:\Gawean Rebinmas\Tree Counting Project\Training Tree Counter Sawit Current\CITRA DRONE DATASET BARU 2025\Divisi 1 A\Sub Divisi Air Kundo 2025.tif"
    SHAPEFILE_PATH = r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\TBM_1A_Kundo.shp"
    
    print("=" * 70)
    print("PALM TREE DETECTION - DATASET PREPARATION")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: Make sure shapefile contains POINT geometry (tree locations)")
    print("   If you have POLYGON, the script will use centroids")
    print("\n📋 CHOOSE STRATEGY:")
    print("   1. Tiled Dataset (Recommended for large images)")
    print("      - Splits large image into 640x640 chips")
    print("      - Good balance of context and detail")
    print("\n   2. Augmented Dataset (Best for small objects)")
    print("      - Multiple zoom levels (1x, 1.5x, 2x)")
    print("      - Helps model learn trees at different scales")
    print("      - More data = better generalization")
    print("=" * 70)
    
    # Initialize preparer
    preparer = PalmTreeDatasetPreparer(IMAGE_PATH, SHAPEFILE_PATH, 
                                       r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\YOLO_Dataset")
    
    # Read data
    preparer.read_image_and_annotations()
    
    # Strategy 1: Tiled dataset (uncomment to use)
    print("\n" + "=" * 70)
    print("STRATEGY 1: Creating tiled dataset...")
    print("=" * 70)
    tiles = preparer.create_tiled_dataset(tile_size=640, overlap=0.2, val_split=0.2)
    
    # Strategy 2: Augmented dataset (uncomment to use)
    # print("\n" + "=" * 70)
    # print("STRATEGY 2: Creating augmented dataset...")
    # print("=" * 70)
    # samples = preparer.create_augmented_dataset(
    #     base_size=640,
    #     zoom_levels=[1.0, 1.5, 2.0],
    #     random_crops=10,
    #     val_split=0.2
    # )
    
    print("\n" + "=" * 70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Dataset location: {preparer.output_dir}")
    print(f"\n🚀 NEXT STEP - Train with YOLOv8:")
    print(f"""
    # Install ultralytics
    pip install ultralytics
    
    # Train model
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # or yolov11n.pt
    model.train(data='{preparer.output_dir / "data.yaml"}', 
                epochs=100, 
                imgsz=640, 
                batch=16,
                name='palm_tree_detection')
    """)


if __name__ == "__main__":
    main()
