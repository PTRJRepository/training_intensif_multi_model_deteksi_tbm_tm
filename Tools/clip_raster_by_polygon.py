"""
Script to clip/crop TIFF raster files based on polygon shapes from a shapefile.
Uses rasterio and geopandas for geospatial operations.
"""

import os
import glob
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import mapping


def clip_tiff_by_polygon(raster_path: str, shapefile_path: str, output_dir: str, 
                         all_touched: bool = True, crop: bool = True):
    """
    Clip a TIFF raster file based on polygon shapes from a shapefile.
    
    Args:
        raster_path: Path to the input TIFF raster file
        shapefile_path: Path to the shapefile containing polygons
        output_dir: Directory to save the clipped output files
        all_touched: If True, all pixels touched by polygons will be included
        crop: If True, crop the output to the extent of the polygons
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the shapefile
    print(f"Reading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    print(f"Found {len(gdf)} polygons in shapefile")
    
    # Read the raster file
    print(f"Reading raster: {raster_path}")
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        print(f"Raster CRS: {raster_crs}")
        print(f"Raster dimensions: {src.width} x {src.height}")
        
        # Check if CRS matches
        if gdf.crs != raster_crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {raster_crs}")
            gdf = gdf.to_crs(raster_crs)
    
    # Clip raster for each polygon
    for idx, row in gdf.iterrows():
        print(f"\nProcessing polygon {idx + 1}/{len(gdf)}...")
        
        # Get the geometry
        geom = [mapping(row.geometry)]
        
        # Get a name for this polygon (use index or any available attribute)
        # Try to find a good name column
        name = None
        for col in ['NAME', 'name', 'Name', 'ID', 'id', 'FID', 'fid']:
            if col in gdf.columns:
                name = str(row[col]).replace(' ', '_').replace('/', '_')
                break
        
        if name is None:
            name = f"polygon_{idx}"
        
        # Clip the raster
        try:
            with rasterio.open(raster_path) as src:
                out_image, out_transform = mask(
                    src, 
                    geom, 
                    crop=crop, 
                    all_touched=all_touched,
                    filled=True,
                    nodata=src.nodata if src.nodata is not None else 0
                )
                
                # Update metadata
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                
                # Save the clipped raster
                output_path = os.path.join(output_dir, f"{name}_clipped.tif")
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                
                print(f"✓ Saved: {output_path}")
                print(f"  Dimensions: {out_image.shape[2]} x {out_image.shape[1]}")
                
        except Exception as e:
            print(f"✗ Error processing polygon {idx}: {str(e)}")
    
    print(f"\n✓ Clipping complete! Output saved to: {output_dir}")


def clip_tiff_by_polygon_merged(raster_path: str, shapefile_path: str, 
                                 output_path: str, all_touched: bool = True, 
                                 crop: bool = True):
    """
    Clip a TIFF raster file using all polygons merged into one mask.
    
    Args:
        raster_path: Path to the input TIFF raster file
        shapefile_path: Path to the shapefile containing polygons
        output_path: Path to save the merged clipped output file
        all_touched: If True, all pixels touched by polygons will be included
        crop: If True, crop the output to the extent of the polygons
    """
    
    # Read the shapefile
    print(f"Reading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    print(f"Found {len(gdf)} polygons in shapefile")
    
    # Read the raster file
    print(f"Reading raster: {raster_path}")
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        
        # Check if CRS matches
        if gdf.crs != raster_crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {raster_crs}")
            gdf = gdf.to_crs(raster_crs)
    
    # Convert all geometries to mappings
    geometries = [mapping(geom) for geom in gdf.geometry]
    
    # Clip the raster with all polygons
    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(
                src, 
                geometries, 
                crop=crop, 
                all_touched=all_touched,
                filled=True,
                nodata=src.nodata if src.nodata is not None else 0
            )
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Save the clipped raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
            print(f"✓ Saved merged clip: {output_path}")
            print(f"  Dimensions: {out_image.shape[2]} x {out_image.shape[1]}")
            
    except Exception as e:
        print(f"✗ Error during clipping: {str(e)}")
        raise


if __name__ == "__main__":
    # Configuration
    RASTER_PATH = r"D:\Gawean Rebinmas\Tree Counting Project\Training Tree Counter Sawit Current\CITRA DRONE DATASET BARU 2025\Divisi 1 A\Sub Divisi Air Kundo 2025.tif"
    SHAPEFILE_PATH = r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\Potong_layer_1A.shp"
    OUTPUT_DIR = r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Processed\Clipped"
    
    # Option 1: Clip raster for each polygon separately
    print("=" * 60)
    print("Option 1: Clipping raster for each polygon separately")
    print("=" * 60)
    
    # Check if raster file exists
    if not os.path.exists(RASTER_PATH):
        print(f"\n⚠ Raster file not found: {RASTER_PATH}")
        print("Please check the path and try again")
    else:
        clip_tiff_by_polygon(RASTER_PATH, SHAPEFILE_PATH, OUTPUT_DIR)
    
    # Option 2: Clip raster with all polygons merged (uncomment to use)
    # print("\n" + "=" * 60)
    # print("Option 2: Clipping raster with all polygons merged")
    # print("=" * 60)
    # OUTPUT_PATH = os.path.join(OUTPUT_DIR, "merged_clipped.tif")
    # clip_tiff_by_polygon_merged(RASTER_PATH, SHAPEFILE_PATH, OUTPUT_PATH)
