"""Dataset preparation for palm-tree detection from multiple images.

Supports the original positive-tile generation workflow and an append-only
background tile mode for adding negative examples into an existing YOLO dataset.
"""

import argparse
import cv2
import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from tqdm import tqdm
import random
import yaml


class MultiImageDatasetPreparer:
    """Prepare dataset for palm tree detection from multiple drone images and shapefiles."""

    def __init__(self, output_dir, image_size=640, seed=42):
        """
        Initialize the dataset preparer.

        Args:
            output_dir: Directory to save the prepared dataset
            image_size: Target image size for training (default: 640 for YOLO)
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.rng = random.Random(seed)

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
        print(f"\n{'=' * 70}")
        print(f"📂 Processing: {name if name else image_path}")
        print(f"{'=' * 70}")

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

            if geom.geom_type == "Point":
                px, py = ~transform * (geom.x, geom.y)
                tree_points.append({"x": int(px), "y": int(py)})
            elif geom.geom_type == "MultiPoint":
                # MultiPoint can contain multiple points
                for point in geom.geoms:
                    if point.geom_type == "Point":
                        px, py = ~transform * (point.x, point.y)
                        tree_points.append({"x": int(px), "y": int(py)})
            elif geom.geom_type in ["Polygon", "MultiPolygon"]:
                centroid = geom.centroid
                px, py = ~transform * (centroid.x, centroid.y)
                tree_points.append({"x": int(px), "y": int(py)})

        print(f"   ✓ Converted {len(tree_points)} points to pixel coordinates")

        # Store for tiling
        self.current_image = image
        self.current_tree_points = tree_points
        self.current_name = name

    def create_tiled_dataset(
        self,
        tile_size=640,
        overlap=0.2,
        include_background=False,
        negative_ratio=0.5,
        background_min_std=12.0,
    ):
        """
        Create tiled dataset from current image-shapefile pair.

        Args:
            tile_size: Size of each tile (default: 640)
            overlap: Overlap between tiles (default: 0.2 = 20%)
        """
        print(f"\n🔲 Creating tiled dataset...")
        print(f"   Tile size: {tile_size}x{tile_size}")
        print(f"   Overlap: {overlap * 100}%")

        step = int(tile_size * (1 - overlap))
        img_height, img_width = self.current_image.shape[:2]

        tiles_count = 0
        negative_candidates = []

        # Generate tiles
        for y in range(0, img_height - tile_size + 1, step):
            for x in range(0, img_width - tile_size + 1, step):
                # Extract tile
                tile_img = self.current_image[y : y + tile_size, x : x + tile_size]

                # Find trees in this tile
                trees_in_tile = []
                for tree in self.current_tree_points:
                    tx, ty = tree["x"] - x, tree["y"] - y
                    if 0 <= tx < tile_size and 0 <= ty < tile_size:
                        # Convert to YOLO format (normalized)
                        x_center = tx / tile_size
                        y_center = ty / tile_size

                        # Estimate bbox size based on tree spacing
                        bbox_size = 10  # Adjust based on your tree pixel size
                        w = bbox_size / tile_size
                        h = bbox_size / tile_size

                        trees_in_tile.append(
                            {
                                "class": 0,
                                "x_center": x_center,
                                "y_center": y_center,
                                "width": w,
                                "height": h,
                            }
                        )

                # Keep positive tiles and optionally sample background tiles later.
                if len(trees_in_tile) > 0:
                    self.all_tiles.append(
                        {
                            "id": self.global_tile_id,
                            "image": tile_img,
                            "trees": trees_in_tile,
                            "source": self.current_name,
                            "is_background": False,
                        }
                    )
                    self.global_tile_id += 1
                    tiles_count += 1
                elif include_background and self._is_background_candidate(
                    tile_img, min_std=background_min_std
                ):
                    negative_candidates.append(tile_img.copy())

        background_count = 0
        if include_background and tiles_count > 0 and negative_candidates:
            target_negative_count = min(
                len(negative_candidates),
                max(1, int(round(tiles_count * float(negative_ratio)))),
            )
            selected_negatives = self.rng.sample(
                negative_candidates, target_negative_count
            )
            for tile_img in selected_negatives:
                self.all_tiles.append(
                    {
                        "id": self.global_tile_id,
                        "image": tile_img,
                        "trees": [],
                        "source": self.current_name,
                        "is_background": True,
                    }
                )
                self.global_tile_id += 1
                background_count += 1

        print(f"   ✓ Generated {tiles_count} tiles from {self.current_name}")
        if include_background:
            print(
                f"   ✓ Added {background_count} background tiles from {self.current_name}"
            )

    @staticmethod
    def _is_background_candidate(tile_img, min_std=12.0, max_zero_ratio=0.75):
        if tile_img.size == 0:
            return False

        non_zero_ratio = float(np.count_nonzero(tile_img)) / float(tile_img.size)
        if non_zero_ratio < (1.0 - max_zero_ratio):
            return False

        if tile_img.ndim == 3 and tile_img.shape[2] >= 3:
            gray = cv2.cvtColor(tile_img[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif tile_img.ndim == 3:
            gray = tile_img[:, :, 0]
        else:
            gray = tile_img

        return float(gray.std()) >= float(min_std)

    def save_dataset(self, val_split=0.2):
        """
        Save all tiles to disk and create YOLO config.

        Args:
            val_split: Fraction for validation (default: 0.2)
        """
        print(f"\n{'=' * 70}")
        print(f"💾 Saving dataset...")
        print(f"{'=' * 70}")

        # Shuffle all tiles
        self.rng.shuffle(self.all_tiles)

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
            tile_id = tile["id"]
            source = tile["source"]
            is_background = bool(tile.get("is_background", False))

            # Save image
            suffix = "bg_tile" if is_background else "tile"
            img_filename = f"{source}_{suffix}_{tile_id:05d}.jpg"
            img_path = self.output_dir / "images" / split / img_filename

            # Handle different number of bands
            if tile["image"].shape[2] == 3:
                cv2.imwrite(
                    str(img_path), cv2.cvtColor(tile["image"], cv2.COLOR_RGB2BGR)
                )
            elif tile["image"].shape[2] == 1:
                cv2.imwrite(str(img_path), tile["image"][:, :, 0])
            else:
                # Take first 3 bands
                cv2.imwrite(
                    str(img_path),
                    cv2.cvtColor(tile["image"][:, :, :3], cv2.COLOR_RGB2BGR),
                )

            # Save label (YOLO format)
            label_filename = f"{source}_{suffix}_{tile_id:05d}.txt"
            label_path = self.output_dir / "labels" / split / label_filename

            with open(label_path, "w") as f:
                for tree in tile["trees"]:
                    # YOLO format: class x_center y_center width height
                    f.write(
                        f"{tree['class']} {tree['x_center']:.6f} {tree['y_center']:.6f} "
                        f"{tree['width']:.6f} {tree['height']:.6f}\n"
                    )

    def _next_background_index(self, source_name):
        max_index = -1
        for split in ("train", "val"):
            for image_path in (self.output_dir / "images" / split).glob(
                f"{source_name}_bg_tile_*.jpg"
            ):
                stem = image_path.stem
                try:
                    max_index = max(max_index, int(stem.rsplit("_", 1)[-1]))
                except ValueError:
                    continue
        return max_index + 1

    @staticmethod
    def _erase_points_from_tile(tile_img, local_points, erase_radius=10):
        if tile_img.ndim != 3:
            return tile_img.copy()

        work_img = tile_img[:, :, :3].copy()
        mask = np.zeros(work_img.shape[:2], dtype=np.uint8)
        for px, py in local_points:
            cv2.circle(mask, (int(px), int(py)), int(erase_radius), 255, thickness=-1)

        if np.count_nonzero(mask) == 0:
            return work_img

        return cv2.inpaint(work_img, mask, 3, cv2.INPAINT_TELEA)

    def append_background_tiles_to_existing_dataset(
        self,
        tile_size=640,
        overlap=0.2,
        negative_ratio=0.5,
        background_min_std=12.0,
        val_split=0.2,
        synthetic_fallback=True,
        erase_radius=10,
    ):
        print(f"\n🟫 Appending background tiles...")
        print(f"   Tile size: {tile_size}x{tile_size}")
        print(f"   Overlap: {overlap * 100}%")
        print(f"   Negative ratio target: {negative_ratio}")

        step = int(tile_size * (1 - overlap))
        img_height, img_width = self.current_image.shape[:2]
        negative_candidates = []
        synthetic_candidates = []

        for y in range(0, img_height - tile_size + 1, step):
            for x in range(0, img_width - tile_size + 1, step):
                local_points = []
                for tree in self.current_tree_points:
                    tx, ty = tree["x"] - x, tree["y"] - y
                    if 0 <= tx < tile_size and 0 <= ty < tile_size:
                        local_points.append((tx, ty))

                tile_img = self.current_image[y : y + tile_size, x : x + tile_size]

                if not local_points:
                    if self._is_background_candidate(
                        tile_img, min_std=background_min_std
                    ):
                        negative_candidates.append(tile_img.copy())
                    continue

                if synthetic_fallback:
                    synthetic_tile = self._erase_points_from_tile(
                        tile_img,
                        local_points,
                        erase_radius=erase_radius,
                    )
                    if self._is_background_candidate(
                        synthetic_tile, min_std=background_min_std
                    ):
                        synthetic_candidates.append(synthetic_tile)

        candidate_pool = negative_candidates
        candidate_label = "background"
        if not candidate_pool and synthetic_fallback:
            candidate_pool = synthetic_candidates
            candidate_label = "synthetic background"

        existing_positive_count = 0
        for split in ("train", "val"):
            existing_positive_count += len(
                list(
                    (self.output_dir / "images" / split).glob(
                        f"{self.current_name}_tile_*.jpg"
                    )
                )
            )

        target_negative_count = (
            min(
                len(candidate_pool),
                max(1, int(round(existing_positive_count * float(negative_ratio)))),
            )
            if existing_positive_count > 0
            else 0
        )

        if target_negative_count == 0:
            print(f"   ! No eligible background tiles found for {self.current_name}")
            return 0

        selected_negatives = self.rng.sample(candidate_pool, target_negative_count)
        next_index = self._next_background_index(self.current_name)

        saved_count = 0
        split_counts = {"train": 0, "val": 0}
        for offset, tile_img in enumerate(
            tqdm(selected_negatives, desc=f"Saving bg {self.current_name}")
        ):
            tile_id = next_index + offset
            split = "val" if self.rng.random() < val_split else "train"
            split_counts[split] += 1

            img_filename = f"{self.current_name}_bg_tile_{tile_id:05d}.jpg"
            img_path = self.output_dir / "images" / split / img_filename
            label_path = (
                self.output_dir
                / "labels"
                / split
                / f"{self.current_name}_bg_tile_{tile_id:05d}.txt"
            )

            if tile_img.shape[2] == 3:
                cv2.imwrite(str(img_path), cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))
            elif tile_img.shape[2] == 1:
                cv2.imwrite(str(img_path), tile_img[:, :, 0])
            else:
                cv2.imwrite(
                    str(img_path), cv2.cvtColor(tile_img[:, :, :3], cv2.COLOR_RGB2BGR)
                )

            label_path.write_text("", encoding="utf-8")
            saved_count += 1

        print(
            f"   ✓ Added {saved_count} {candidate_label} tiles for {self.current_name}"
        )
        print(f"   ↳ train: {split_counts['train']} | val: {split_counts['val']}")
        return saved_count

    def _create_yolo_config(self):
        """Create YOLO training configuration file."""
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "TBM"},
        }

        config_path = self.output_dir / "data.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"\n📝 Created YOLO config: {config_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare positive and background tiles for palm-tree detection"
    )
    parser.add_argument(
        "--append-background-only",
        action="store_true",
        help="Append background tiles into the existing dataset without regenerating positives",
    )
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size")
    parser.add_argument("--overlap", type=float, default=0.2, help="Tile overlap")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.5,
        help="Background-to-positive ratio target",
    )
    parser.add_argument(
        "--background-min-std",
        type=float,
        default=12.0,
        help="Minimum grayscale std-dev for a valid background tile",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    """Main function - prepare dataset from multiple image-shapefile pairs."""
    args = parse_args()

    print("=" * 70)
    print("🌴 PALM TREE DETECTION - MULTI-IMAGE DATASET PREPARATION")
    print("=" * 70)
    print("\n📊 This script combines multiple images and shapefiles into one dataset")
    print("   Strategy: Tiled chips (640x640) for small object detection")
    print("=" * 70)

    # Initialize preparer
    output_dir = r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\dataset_pakai_ini"
    preparer = MultiImageDatasetPreparer(output_dir, image_size=640, seed=args.seed)

    # Add image-shapefile pair (using new raster: Sub Divisi Air Kundo high resolution)
    preparer.add_image_shapefile_pair(
        image_path=r"D:\Gawean Rebinmas\Tree Counting Project\Training Tree Counter Sawit Current\CITRA DRONE DATASET BARU 2025\Divisi 1 A\Sub Divisi Air kundo_high.tif",
        shapefile_path=r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\TBM_1A_Kundo.shp",
        name="Atha_Kundo",
    )
    if args.append_background_only:
        preparer.append_background_tiles_to_existing_dataset(
            tile_size=args.tile_size,
            overlap=args.overlap,
            negative_ratio=args.negative_ratio,
            background_min_std=args.background_min_std,
            val_split=args.val_split,
        )
    else:
        preparer.create_tiled_dataset(
            tile_size=args.tile_size,
            overlap=args.overlap,
            include_background=True,
            negative_ratio=args.negative_ratio,
            background_min_std=args.background_min_std,
        )

    # Save dataset only when regenerating positives.
    if not args.append_background_only:
        preparer.save_dataset(val_split=args.val_split)

    print(f"\n{'=' * 70}")
    print("✅ DATASET PREPARATION COMPLETE!")
    print(f"{'=' * 70}")
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
