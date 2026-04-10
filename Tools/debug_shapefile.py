"""Debug script to check shapefile geometry and CRS issues."""

import geopandas as gpd
import rasterio

# Check first pair
print("="*70)
print("IMAGE 1: Dataset_1A_Atha.tif")
print("="*70)
with rasterio.open(r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\Processed\Clipped\Dataset_1A_Atha.tif") as src:
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Image size: {src.width} x {src.height}")

gdf1 = gpd.read_file(r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\TBM_1A_Kundo.shp")
print(f"\nShapefile 1: TBM_1A_Kundo.shp")
print(f"  CRS: {gdf1.crs}")
print(f"  Total features: {len(gdf1)}")
print(f"  Geometry types: {gdf1.geometry.geom_type.value_counts().to_dict()}")
print(f"  Bounds: {gdf1.total_bounds}")

# Check second pair
print("\n" + "="*70)
print("IMAGE 2: tbm_1a_isnin.tif")
print("="*70)
with rasterio.open(r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\Processed\Clipped\tbm_1a_isnin.tif") as src:
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Image size: {src.width} x {src.height}")

gdf2 = gpd.read_file(r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\TBM\TBM.shp")
print(f"\nShapefile 2: TBM.shp")
print(f"  CRS: {gdf2.crs}")
print(f"  Total features: {len(gdf2)}")
print(f"  Non-null geometries: {gdf2.geometry.notna().sum()}")
print(f"  Geometry types: {gdf2.geometry[gdf2.geometry.notna()].geom_type.value_counts().to_dict()}")
print(f"  Bounds: {gdf2[gdf2.geometry.notna()].total_bounds}")

# Test reprojection
print("\n" + "="*70)
print("TEST REPROJECTION")
print("="*70)
with rasterio.open(r"D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model\Dataset\Raw Dataset\TBM\Processed\Clipped\tbm_1a_isnin.tif") as src:
    image_crs = src.crs
    transform = src.transform
    
    if gdf2.crs != image_crs:
        print(f"Reprojecting from {gdf2.crs} to {image_crs}")
        gdf2_reproj = gdf2.to_crs(image_crs)
        print(f"  Reprojected bounds: {gdf2_reproj[gdf2_reproj.geometry.notna()].total_bounds}")
        
        # Test one point
        valid_geom = gdf2_reproj[gdf2_reproj.geometry.notna()].geometry.iloc[0]
        if valid_geom.geom_type == 'Point':
            px, py = ~transform * (valid_geom.x, valid_geom.y)
            print(f"  Test point -> pixel: ({px}, {py})")
            print(f"  In image bounds: {0 <= px < src.width and 0 <= py < src.height}")
