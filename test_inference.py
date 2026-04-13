"""
Test inference with flexible window sizes on a sample TIFF image.
"""

import os
import sys
import rasterio
import numpy as np
from PIL import Image
from io import BytesIO
import time

# Add the auto_labeling module to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "auto_labeling", "python_app")
)

from core.detector import TreeDetector

# Paths
BASE_DIR = (
    r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model"
)
MODEL_PATH = os.path.join(
    BASE_DIR, "Train", "runs", "exp2_yolo11s_1280", "weights", "best.pt"
)
TEST_IMAGE = r"D:\Gawean Rebinmas\Tree Counting Project\Training Tree Counter Sawit Current\CITRA DRONE DATASET BARU 2025\Divisi 2 A\Sub Divisi Gunung Panjang 8-9-2025.tif"


def test_single_slice(detector, tiff_path, x, y, w, h, imgsz=640):
    """Test detection on a single slice."""
    print(f"\n--- Testing slice at ({x}, {y}) size {w}x{h} with imgsz={imgsz} ---")

    with rasterio.open(tiff_path) as src:
        window = rasterio.windows.Window(x, y, w, h)
        num_bands = min(src.count, 3)
        data = src.read(list(range(1, num_bands + 1)), window=window)

        if num_bands == 1:
            data = np.repeat(data, 3, axis=0)
        elif num_bands == 2:
            data = np.concatenate([data, np.zeros((1, h, w), dtype=data.dtype)], axis=0)
        data = np.transpose(data, (1, 2, 0))

        if data.dtype != np.uint8:
            sample = data[::4, ::4]
            p2, p98 = np.percentile(sample, (2, 98))
            if p98 > p2:
                data = np.clip(data, p2, p98)
                data = ((data - p2) / (p98 - p2) * 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)

    start = time.time()
    points = detector.predict_image(data, conf=0.1, imgsz=imgsz)
    elapsed = time.time() - start

    # Adjust points to absolute coordinates
    points_abs = [{"x": p["x"] + x, "y": p["y"] + y, "conf": p["conf"]} for p in points]

    print(f"  Found {len(points)} detections in {elapsed:.2f}s")
    print(f"  Slice mean: {data.mean():.1f}")
    return points_abs


def test_full_scan(detector, tiff_path, tile_size=640, overlap=0.25, imgsz=640):
    """Test SAHI-style full scan with flexible parameters."""
    print(
        f"\n--- Testing full scan: tile_size={tile_size}, overlap={overlap}, imgsz={imgsz} ---"
    )

    # Use a small window for testing (2000x2000 from top-left)
    scan_window = [0, 0, 2000, 2000]

    start = time.time()
    count = 0
    all_points = []

    for update in detector.predict_tiff_full(
        tiff_path,
        conf=0.1,
        tile_size=tile_size,
        overlap=overlap,
        scan_window=scan_window,
        imgsz=imgsz,
    ):
        if update["type"] == "window":
            count += 1
        elif update["type"] == "points":
            all_points.extend(update["points"])
            print(
                f"  Progress: {update.get('progress', 0) * 100:.1f}% - Total points so far: {len(all_points)}"
            )
        elif update["type"] == "final":
            all_points = update["points"]

    elapsed = time.time() - start

    print(f"\n  Total tiles processed: {count}")
    print(f"  Total detections (after merge): {len(all_points)}")
    print(f"  Time: {elapsed:.2f}s")
    return all_points


def main():
    print(f"Model: {MODEL_PATH}")
    print(f"Test image: {TEST_IMAGE}")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(TEST_IMAGE):
        print(f"ERROR: Test image not found at {TEST_IMAGE}")
        return

    # Initialize detector
    print("\nInitializing detector...")
    detector = TreeDetector(MODEL_PATH)

    # Get image info
    with rasterio.open(TEST_IMAGE) as src:
        print(f"Image size: {src.width} x {src.height}")

    # Test 1: Single slice with different imgsz values
    print("\n" + "=" * 60)
    print("TEST 1: Single slice with flexible inference size")
    print("=" * 60)

    # Test different inference sizes
    for imgsz in [640, 1280, 1920]:
        try:
            points = test_single_slice(
                detector, TEST_IMAGE, 0, 0, 640, 640, imgsz=imgsz
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    # Test 2: Different window sizes
    print("\n" + "=" * 60)
    print("TEST 2: Different window sizes with fixed imgsz=640")
    print("=" * 60)

    for window_size in [320, 640, 1280]:
        try:
            points = test_single_slice(
                detector, TEST_IMAGE, 0, 0, window_size, window_size, imgsz=640
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    # Test 3: Full scan on small area
    print("\n" + "=" * 60)
    print("TEST 3: SAHI-style full scan (2000x2000 area)")
    print("=" * 60)

    for tile_size in [320, 640]:
        for imgsz in [640, 1280]:
            try:
                test_full_scan(
                    detector, TEST_IMAGE, tile_size=tile_size, overlap=0.25, imgsz=imgsz
                )
            except Exception as e:
                print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
