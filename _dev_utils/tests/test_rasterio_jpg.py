import rasterio
import os

img_path = r"Dataset/dataset_pakai_ini/images/train/Atha_Kundo_tile_00000.jpg"
if not os.path.exists(img_path):
    print(f"File not found: {img_path}")
else:
    try:
        with rasterio.open(img_path) as src:
            print(f"Successfully opened {img_path}")
            print(f"Width: {src.width}, Height: {src.height}, Count: {src.count}")
    except Exception as e:
        print(f"Failed to open {img_path} with rasterio: {e}")
