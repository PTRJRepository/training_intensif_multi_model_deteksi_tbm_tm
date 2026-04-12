import os
import json
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Body
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import rasterio
from io import BytesIO
from PIL import Image
import numpy as np

from core.detector import TreeDetector
from core.exporter import Exporter

app = FastAPI(title="Tree Counting Auto-Labeling API")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Specific dataset images
SPECIFIC_IMAGES = [
    r"D:\Gawean Rebinmas\Tree Counting Project\Training Tree Counter Sawit Current\CITRA DRONE DATASET BARU 2025\Divisi 1 A\Sub Divisi Air Kundo 2025.tif",
    r"D:\Gawean Rebinmas\Tree Counting Project\Training Tree Counter Sawit Current\CITRA DRONE DATASET BARU 2025\Divisi 1 B\SUB DIVISI GUNUNG RUM 12-09-2025.tif",
    r"D:\Gawean Rebinmas\Tree Counting Project\Training Tree Counter Sawit Current\CITRA DRONE DATASET BARU 2025\Divisi 1 B\Sub Divisi Padang Tembalun 08 09 2025.tif",
    r"D:\Gawean Rebinmas\Tree Counting Project\Training Tree Counter Sawit Current\CITRA DRONE DATASET BARU 2025\Divisi 2 A\P 2501.tif",
]

MODEL_PATH = os.path.join(BASE_DIR, "Train", "runs", "exp1_yolo11n_640", "weights", "best.pt")

# Fallback to any available trained model
if not os.path.exists(MODEL_PATH):
    import glob
    candidates = glob.glob(os.path.join(BASE_DIR, "Train", "runs", "**", "weights", "best.pt"), recursive=True)
    if candidates:
        MODEL_PATH = candidates[0]

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
os.makedirs(SAVE_DIR, exist_ok=True)

# Global Detector
detector = TreeDetector(MODEL_PATH)


@app.get("/api/images")
async def list_images():
    """List only the specific TIFF images configured."""
    images = []
    seen_paths = set()

    for full_path in SPECIFIC_IMAGES:
        if not os.path.exists(full_path):
            print(f"[WARN] File not found: {full_path}")
            continue
        full_path = os.path.abspath(full_path)
        if full_path not in seen_paths:
            images.append({
                "name": os.path.basename(full_path),
                "path": full_path
            })
            seen_paths.add(full_path)

    print(f"[INFO] Found {len(images)} images")
    return images


def get_full_path(provided_path: str):
    """Utility to resolve paths safely - only allows SPECIFIC_IMAGES."""
    if os.path.isabs(provided_path):
        if provided_path in SPECIFIC_IMAGES and os.path.exists(provided_path):
            return provided_path
        raise HTTPException(status_code=403, detail="Path not allowed")
    raise HTTPException(status_code=404, detail="Image not found")

from fastapi.responses import FileResponse, StreamingResponse, Response
import functools

# Simple in-memory cache for image info
@functools.lru_cache(maxsize=100)
def get_cached_image_info(path: str):
    with rasterio.open(path) as src:
        bounds = src.bounds
        return {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "bands": src.count,
            "overviews": src.overviews(1)
        }

@app.get("/api/image-info")
async def get_image_info(path: str):
    try:
        full_path = get_full_path(path)
        return get_cached_image_info(full_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tile")
async def get_tile(path: str, x: int, y: int, w: int, h: int, scale: float = 1.0):
    """Returns an optimized WebP crop of the image."""
    full_path = get_full_path(path)
    
    with rasterio.open(full_path) as src:
        actual_w = min(w, src.width - x)
        actual_h = min(h, src.height - y)
        
        if actual_w <= 0 or actual_h <= 0:
            raise HTTPException(status_code=400, detail="Invalid dimensions")

        # Calculate output size based on scale
        out_w = max(1, int(actual_w * scale))
        out_h = max(1, int(actual_h * scale))
        
        window = rasterio.windows.Window(x, y, actual_w, actual_h)
        
        # Optimization: Use rasterio's internal decimation (uses overviews if available)
        # This is MUCH faster than reading full res and resizing manually
        num_bands = min(src.count, 3)
        data = src.read(
            list(range(1, num_bands + 1)), 
            window=window,
            out_shape=(num_bands, out_h, out_w),
            resampling=rasterio.enums.Resampling.bilinear if scale < 1.0 else rasterio.enums.Resampling.nearest
        )
        
        if num_bands == 1:
            data = np.repeat(data, 3, axis=0)
        elif num_bands == 2:
            data = np.concatenate([data, np.zeros((1, out_h, out_w), dtype=data.dtype)], axis=0)
            
        data = np.transpose(data, (1, 2, 0))
        
        if data.dtype != np.uint8:
            # Smart normalization: use a fixed range or dynamic based on sample
            data = ((data - data.min()) / (data.max() - data.min() + 1e-5) * 255).astype(np.uint8)
        
        img = Image.fromarray(data)
        img_buffer = BytesIO()
        # Use WebP for better compression/speed
        img.save(img_buffer, format="WebP", quality=80)
        return Response(content=img_buffer.getvalue(), media_type="image/webp")


@app.post("/api/detect")
async def run_detection(path: str, x: int, y: int, w: int, h: int):
    """Run detection on a specific window."""
    full_path = get_full_path(path)
    print(f"[DETECT] path={path}, x={x}, y={y}, w={w}, h={h}")
    try:
        points = detector.predict_tiff_slice(full_path, x, y, w, h)
        print(f"[DETECT] Found {len(points)} points")
        return {"points": points}
    except Exception as e:
        print(f"[DETECT] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-full")
async def run_full_detection(path: str, conf: float = 0.1, tile_size: int = 640, overlap: float = 0.25, window: str = None, polygon: str = None):
    """
    Streaming SAHI-style full image or ROI detection (box or polygon).
    window: JSON string of [x, y, w, h]
    polygon: JSON string of [[x1, y1], [x2, y2], ...]
    """
    full_path = get_full_path(path)
    
    scan_window = None
    if window:
        try: scan_window = json.loads(window)
        except: pass

    scan_polygon = None
    if polygon:
        try: scan_polygon = json.loads(polygon)
        except: pass

    # If overlap is > 1, assume it's pixels. If < 1, assume percentage.
    actual_overlap = overlap
    if overlap >= 1:
        actual_overlap = overlap / tile_size
        
    print(f"[DETECT-STREAM] path={path}, window={scan_window}, poly={len(scan_polygon) if scan_polygon else 0}")

    async def event_generator():
        try:
            for update in detector.predict_tiff_full(full_path, conf=conf, tile_size=tile_size, overlap=actual_overlap, scan_window=scan_window, polygon=scan_polygon):
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            print(f"[STREAM-ERROR] {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )

@app.post("/api/save")
async def save_labels(path: str, data: list = Body(...)):
    """Save points to a local JSON file."""
    fname = path.replace(os.sep, "_").replace("/", "_") + ".json"
    save_path = os.path.join(SAVE_DIR, fname)
    with open(save_path, "w") as f:
        json.dump(data, f)
    return {"status": "success", "file": fname}

@app.get("/api/load")
async def load_labels(path: str):
    """Load points from a local JSON file."""
    fname = path.replace(os.sep, "_").replace("/", "_") + ".json"
    save_path = os.path.join(SAVE_DIR, fname)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            return json.load(f)
    return []

@app.post("/api/export/shapefile")
async def export_shapefile(path: str, labels: list = Body(...)):
    """Export to Shapefile."""
    full_path = get_full_path(path)
    base_name = os.path.basename(path).split('.')[0]
    # Use timestamp to avoid overwriting
    import time
    output_name = f"{base_name}_labels_{int(time.time())}.shp"
    output_path = os.path.join(SAVE_DIR, output_name)

    Exporter.to_shapefile(labels, full_path, output_path)
    return {"status": "success", "path": output_path, "base_name": base_name}

@app.post("/api/export/dataset")
async def export_dataset(path: str, labels: list = Body(...)):
    """Export to YOLO dataset."""
    full_path = get_full_path(path)
    output_dir = os.path.join(SAVE_DIR, "dataset_" + os.path.basename(path).split('.')[0])
    Exporter.to_yolo_dataset(labels, full_path, output_dir)
    return {"status": "success", "directory": output_dir}

# Serve static files for React (if built)
if os.path.exists(os.path.join(os.path.dirname(__file__), "static")):
    app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
