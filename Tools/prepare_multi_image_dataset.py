"""
Dataset preparation for small object detection (palm tree counting from drone imagery).
Supports MULTIPLE images and shapefiles for training.
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
import yaml


class MultiImageDatasetPreparer:
    """Prepare dataset for palm tree detection from multiple drone images and shapefiles."""

    def __init__(self, output_dir, image_size=640):
        """
        Initialize the dataset preparer.
        
        Args:
            output_dir: Directory to save the prepared dataset
            image_size: Target image size for training (default: 640 for YOLO)
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        
        # Create output directories
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        self.all_tiles = []
        self.global_tile_id = 0
        
    def add_image_shapefile_pair(self, image_path, shapefile_path, name=""):
        """
        Add an image-shapefile pair to the dataset.
        
        Args:
            image_path: Path to the drone TIFF image
            shapefile_path: Path to shapefile with tree locations
            name: Name identifier for this pair
        """
        print(f"\n{'='*70}")
        print(f"📂 Processing: {name if name else image_path}")
        print(f"{'='*70}")
        
        # Read image
        print("📖 Reading TIFF image...")
        with rasterio.open(image_path) as src:
            image = src.read()
            transform = src.transform
            crs = src.crs
            print(f"   Image shape: {image.shape}")
        
        # Transpose to (height, width, bands)
        if image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
        
        # Read shapefile
        print("📖 Reading shapefile annotations...")
        gdf = gpd.read_file(shapefile_path)
        print(f"   Found {len(gdf)} tree annotations")
        
        # Reproject if needed
        if gdf.crs != crs:
            print(f"   Reprojecting from {gdf.crs} to {crs}")
            gdf = gdf.to_crs(crs)
        
        # Convert to pixel coordinates
        tree_points = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Skip null geometries
            if geom is None:
                continue
                
            if geom.geom_type == 'Point':
                px, py = ~transform * (geom.x, geom.y)
                tree_points.append({'x': int(px), 'y': int(py)})
            elif geom.geom_type == 'MultiPoint':
                # MultiPoint can contain multiple points
                for point in geom.geoms:
                    if point.geom_type == 'Point':
                        px, py = ~transform * (point.x, point.y)
                        tree_points.append({'x': int(px), 'y': int(py)})
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                centroid = geom.centroid
                px, py = ~transform * (centroid.x, centroid.y)
                tree_points.append({'x': int(px), 'y': int(py)})
        
        print(f"   ✓ Converted {len(tree_points)} points to pixel coordinates")
        
        # Store for tiling
        self.current_image = image
        self.current_tree_points = tree_points
        self.current_name = name
        
    def create_tiled_dataset(self, tile_size=640, overlap=0.2):
        """
        Create tiled dataset from current image-shapefile pair.
        
        Args:
            tile_size: Size of each tile (default: 640)
            overlap: Overlap between tiles (default: 0.2 = 20%)
        """
        print(f"\n🔲 Creating tiled dataset...")
        print(f"   Tile size: {tile_size}x{tile_size}")
        print(f"   Overlap: {overlap*100}%")
        
        step = int(tile_size * (1 - overlap))
        img_height, img_width = self.current_image.shape[:2]
        
        tiles_count = 0
        
        # Generate tiles
        for y in range(0, img_height - tile_size + 1, step):
            for x in range(0, img_width - tile_size + 1, step):
                # Extract tile
                tile_img = self.current_image[y:y+tile_size, x:x+tile_size]
                
                # Find trees in this tile
                trees_in_tile = []
                for tree in self.current_tree_points:
                    tx, ty = tree['x'] - x, tree['y'] - y
                    if 0 <= tx < tile_size and 0 <= ty < tile_size:
                        # Convert to YOLO format (normalized)
                        x_center = tx / tile_size
                        y_center = ty / tile_size
                        
                        # Estimate bbox size based on tree spacing
                        bbox_size = 10  # Adjust based on your tree pixel size
                        w = bbox_size / tile_size
                        h = bbox_size / tile_size
                        
                        trees_in_tile.append({
                            'class': 0,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': w,
                            'height': h
                        })
                
                # Only keep tiles with at least one tree
                if len(trees_in_tile) > 0:
                    self.all_tiles.append({
                        'id': self.global_tile_id,
                        'image': tile_img,
                        'trees': trees_in_tile,
                        'source': self.current_name
                    })
                    self.global_tile_id += 1
                    tiles_count += 1
        
        print(f"   ✓ Generated {tiles_count} tiles from {self.current_name}")
        
    def save_dataset(self, val_split=0.2):
        """
        Save all tiles to disk and create YOLO config.
        
        Args:
            val_split: Fraction for validation (default: 0.2)
        """
        print(f"\n{'='*70}")
        print(f"💾 Saving dataset...")
        print(f"{'='*70}")
        
        # Shuffle all tiles
        random.shuffle(self.all_tiles)
        
        # Split train/val
        val_count = max(1, int(len(self.all_tiles) * val_split))
        train_tiles = self.all_tiles[val_count:]
        val_tiles = self.all_tiles[:val_count]
        
        print(f"   Total tiles: {len(self.all_tiles)}")
        print(f"   Train tiles: {len(train_tiles)}")
        print(f"   Val tiles: {len(val_tiles)}")
        
        # Save tiles
        self._save_tiles(train_tiles, "train")
        self._save_tiles(val_tiles, "val")
        
        # Create YOLO config
        self._create_yolo_config()
        
    def _save_tiles(self, tiles, split):
        """Save tiles to disk in YOLO format."""
        print(f"\n💾 Saving {split} tiles...")
        
        for tile in tqdm(tiles, desc=f"Saving {split}"):
            tile_id = tile['id']
            source = tile['source']
            
            # Save image
            img_filename = f"{source}_tile_{tile_id:05d}.jpg"
            img_path = self.output_dir / "images" / split / img_filename
            
            # Handle different number of bands
            if tile['image'].shape[2] == 3:
                cv2.imwrite(str(img_path), cv2.cvtColor(tile['image'], cv2.COLOR_RGB2BGR))
            elif tile['image'].shape[2] == 1:
                cv2.imwrite(str(img_path), tile['image'][:, :, 0])
            else:
                # Take first 3 bands
                cv2.imwrite(str(img_path), cv2.cvtColor(tile['image'][:, :, :3], cv2.COLOR_RGB2BGR))
            
            # Save label (YOLO format)
            label_filename = f"{source}_tile_{tile_id:05d}.txt"
            label_path = self.output_dir / "labels" / split / label_filename
            
            with open(label_path, 'w') as f:
                for tree in tile['trees']:
                    # YOLO format: class x_center y_center width height
                    f.write(f"{tree['class']} {tree['x_center']:.6f} {tree['y_center']:.6f} "
                           f"{tree['width']:.6f} {tree['height']:.6f}\n")
    
    def _create_yolo_config(self):
        """Create YOLO training configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'TBM'
            }
        }
        
        config_path = self.output_dir / "data.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n📝 Created YOLO config: {config_path}")


def main():
    """Main function - prepare dataset from multiple image-shapefile pairs."""
    
    print("=" * 70)
    print("🌴 PALM TREE DETECTION - MULTI-IMAGE DATASET PREPARATION")
    print("=" * 70)
    print("\n📊 This script combines multiple images and shapefiles into one dataset")
    print("   Strategy: Tiled chips (640x640) for small object detection")
    print("=" * 70)
    
    # Initialize preparer
    output_dir = r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\dataset_pakai_ini"
    preparer = MultiImageDatasetPreparer(output_dir, image_size=640)
    
    # Add first image-shapefile pair
    preparer.add_image_shapefile_pair(
        image_path=r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\Processed\Clipped\Dataset_1A_Atha.tif",
        shapefile_path=r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\TBM_1A_Kundo.shp",
        name="Atha_Kundo"
    )
    preparer.create_tiled_dataset(tile_size=640, overlap=0.2)
    
    # Add second image-shapefile pair
    preparer.add_image_shapefile_pair(
        image_path=r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\Processed\Clipped\tbm_1a_isnin.tif",
        shapefile_path=r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\TBM\TBM.shp",
        name="Isnin_TBM"
    )
    preparer.create_tiled_dataset(tile_size=640, overlap=0.2)
    
    # Save dataset
    preparer.save_dataset(val_split=0.2)
    
    print(f"\n{'='*70}")
    print("✅ DATASET PREPARATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📂 Dataset location: {output_dir}")
    print(f"\n📊 Dataset structure:")
    print(f"   {output_dir}/")
    print(f"   ├── images/train/     - Training images")
    print(f"   ├── images/val/       - Validation images")
    print(f"   ├── labels/train/     - Training labels (YOLO format)")
    print(f"   ├── labels/val/       - Validation labels (YOLO format)")
    print(f"   └── data.yaml         - YOLO config file")
    print(f"\n🚀 NEXT STEP - Train with YOLOv8/v11:")
    print(f"""
    # Install ultralytics
    pip install ultralytics
    
    # Train model
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # or yolov11n.pt
    model.train(data='{output_dir}/data.yaml', 
                epochs=100, 
                imgsz=640, 
                batch=16,
                name='palm_tree_detection')
    """)


if __name__ == "__main__":
    main()
