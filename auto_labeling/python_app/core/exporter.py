import os
import json
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from PIL import Image
import numpy as np

class Exporter:
    @staticmethod
    def to_shapefile(points, tiff_path, output_path):
        """
        Convert pixel points to a Shapefile using the TIFF's georeferencing.
        """
        if not points:
            raise ValueError("No points to export")

        try:
            with rasterio.open(tiff_path) as src:
                crs = src.crs
                
                world_coords = []
                for p in points:
                    # p['x'] is column (col), p['y'] is row (row)
                    # rasterio.xy returns (lon, lat) or (east, north)
                    try:
                        lon, lat = src.xy(p['y'], p['x'])
                        world_coords.append((lon, lat))
                    except:
                        # Fallback if transformation fails
                        world_coords.append((p['x'], -p['y']))
                
                # Create GeoDataFrame
                geometry = [Point(xy) for xy in world_coords]
                
                data = {
                    'id': range(len(points)),
                    'label': [p.get('label', 'palm_tree') for p in points],
                    'conf': [round(float(p.get('conf', 1.0)), 4) for p in points],
                    'pix_x': [int(p['x']) for p in points],
                    'pix_y': [int(p['y']) for p in points]
                }
                
                # Use a default CRS if none found to prevent shapefile errors
                if crs is None:
                    print("[WARN] No CRS found in TIFF, using local coordinates.")
                
                gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save to shapefile
                gdf.to_file(output_path)
                print(f"[EXPORT] Saved {len(gdf)} points to {output_path}")
                return output_path
        except Exception as e:
            print(f"[EXPORT-ERROR] {str(e)}")
            raise e

    @staticmethod
    def to_yolo_dataset(points, tiff_path, output_dir, tile_size=640, overlap=0):
        """
        Slices the TIFF into patches and saves them as a YOLO dataset.
        """
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

        with rasterio.open(tiff_path) as src:
            width, height = src.width, src.height
            
            for y in range(0, height, tile_size - overlap):
                for x in range(0, width, tile_size - overlap):
                    w = min(tile_size, width - x)
                    h = min(tile_size, height - y)
                    if w < tile_size // 2 or h < tile_size // 2: continue
                    
                    window = rasterio.windows.Window(x, y, w, h)
                    
                    tile_points = []
                    for p in points:
                        if x <= p['x'] < x + w and y <= p['y'] < y + h:
                            rel_x = (p['x'] - x) / w
                            rel_y = (p['y'] - y) / h
                            tile_points.append((rel_x, rel_y))
                    
                    if not tile_points: continue
                    
                    data = src.read([1, 2, 3], window=window)
                    tile_name = f"{os.path.basename(tiff_path).split('.')[0]}_{x}_{y}"
                    img_path = os.path.join(output_dir, "images", f"{tile_name}.jpg")
                    img_data = np.transpose(data, (1, 2, 0))
                    Image.fromarray(img_data).save(img_path, quality=95)
                    
                    label_path = os.path.join(output_dir, "labels", f"{tile_name}.txt")
                    with open(label_path, "w") as f:
                        for px, py in tile_points:
                            f.write(f"0 {px:.6f} {py:.6f} 0.050000 0.050000\n")
                            
        return output_dir
