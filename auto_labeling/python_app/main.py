import os
import json
import tempfile
import zipfile
import glob
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Body, Request
from typing import Any, Dict, Iterable, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, Response, HTMLResponse
import rasterio
from io import BytesIO
from PIL import Image
import numpy as np
from urllib.parse import quote

from core.detector import MultiModelDetector, TreeDetector
from core.exporter import Exporter

app = FastAPI(title="Tree Counting Auto-Labeling API")


DEBUG_ENABLED = os.environ.get("AUTO_LABEL_DEBUG", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@app.on_event("startup")
async def startup_sync_frontend_build() -> None:
    active_static_dir = get_active_static_dir()
    print(
        f"[FRONTEND] debug={'on' if DEBUG_ENABLED else 'off'} | frontend_dist={FRONTEND_DIST_DIR} | serving={active_static_dir or 'none'}"
    )


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path or ""
    if path == "/" or path.endswith((".html", ".js", ".css", ".map", ".svg")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON_APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_LABELING_DIR = os.path.dirname(PYTHON_APP_DIR)
FRONTEND_DIR = os.path.join(AUTO_LABELING_DIR, "frontend")
FRONTEND_DIST_DIR = os.path.join(FRONTEND_DIR, "dist")
MODEL_CONFIG_DIR = os.path.join(PYTHON_APP_DIR, "config")
MODEL_CONFIG_PATH = os.path.join(MODEL_CONFIG_DIR, "models.json")

# Image search directory (strict source)
DATASET_IMAGE_DIR = r"D:\Gawean Rebinmas\Tree Counting Project\Training Tree Counter Sawit Current\CITRA DRONE DATASET BARU 2025"
IMAGE_DIRECTORIES = [DATASET_IMAGE_DIR]
WHITELIST_PATH = os.path.join(MODEL_CONFIG_DIR, "image_whitelist.json")

# Default export directory (user can change this via API)
DEFAULT_EXPORT_DIR = os.path.join(PYTHON_APP_DIR, "storage", "exports")
os.makedirs(DEFAULT_EXPORT_DIR, exist_ok=True)
os.makedirs(MODEL_CONFIG_DIR, exist_ok=True)

# Cache for discovered images
_discovered_images = None
SUPPORTED_INFER_CLASSES = {"tbm", "tm"}


def debug_log(scope: str, message: str) -> None:
    if DEBUG_ENABLED:
        print(f"[DEBUG:{scope}] {message}")


def get_active_static_dir() -> Optional[str]:
    if os.path.exists(os.path.join(FRONTEND_DIST_DIR, "index.html")):
        return FRONTEND_DIST_DIR
    return None


def _asset_version(path: str) -> str:
    try:
        return str(int(os.path.getmtime(path)))
    except OSError:
        return "0"


def _build_index_html(static_dir: str) -> str:
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as file:
        html = file.read()

    asset_candidates = [
        os.path.join(static_dir, "assets"),
        static_dir,
    ]
    version = max(_asset_version(path) for path in asset_candidates)

    html = re.sub(
        r'((?:src|href)="\./[^"?]+)(")',
        lambda match: f'{match.group(1)}?v={quote(version)}{match.group(2)}',
        html,
    )
    return html


def _load_whitelist_set():
    if not os.path.exists(WHITELIST_PATH):
        return None

    try:
        with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read whitelist: {e}")
        return None

    if not isinstance(raw, list) or not raw:
        return None

    normalized = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        v = item.strip()
        if not v:
            continue
        normalized.add(v.replace("\\", "/").lower())
    return normalized or None


def discover_images():
    """Scan directories for TIFF files recursively and group by folder."""
    global _discovered_images
    if _discovered_images is not None:
        return _discovered_images

    _discovered_images = []
    seen_paths = set()

    # Also build folder structure for grouped display
    folder_groups = {}

    whitelist_set = _load_whitelist_set()

    for directory in IMAGE_DIRECTORIES:
        if not os.path.exists(directory):
            print(f"[WARN] Directory not found: {directory}")
            continue

        # Get folder name for grouping
        folder_name = os.path.basename(directory)

        # Recursively find all .tif and .tiff files
        for tiff_path in glob.glob(
            os.path.join(directory, "**", "*.tif*"), recursive=True
        ):
            tiff_path = os.path.abspath(tiff_path)
            if tiff_path not in seen_paths:
                rel_path = os.path.relpath(tiff_path, directory)
                folder_key = os.path.dirname(rel_path).replace(os.sep, "/")

                if whitelist_set is not None:
                    rel_norm = rel_path.replace("\\", "/").lower()
                    name_norm = os.path.basename(tiff_path).lower()
                    abs_norm = tiff_path.replace("\\", "/").lower()
                    if (
                        rel_norm not in whitelist_set
                        and name_norm not in whitelist_set
                        and abs_norm not in whitelist_set
                    ):
                        continue

                _discovered_images.append(
                    {
                        "name": os.path.basename(tiff_path),
                        "path": tiff_path,
                        "folder": folder_key if folder_key != "." else folder_name,
                        "group": folder_name,
                    }
                )
                seen_paths.add(tiff_path)

    print(f"[INFO] Discovered {len(_discovered_images)} images")
    return _discovered_images


def _to_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "model"


def _normalize_class_tag(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in SUPPORTED_INFER_CLASSES:
        return raw
    return "tbm"


def _format_model_mtime(model_path: str) -> tuple[Optional[str], Optional[float]]:
    if not os.path.isfile(model_path):
        return None, None
    try:
        ts = float(os.path.getmtime(model_path))
    except OSError:
        return None, None
    return (
        # Local server timezone ISO-like string for UI display
        datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
        ts,
    )


def _resolve_model_path(model_path: str) -> str:
    model_path = str(model_path).strip()
    if not model_path:
        return ""
    if os.path.isabs(model_path):
        return os.path.abspath(model_path)
    return os.path.abspath(os.path.join(BASE_DIR, model_path))


def _discover_model_candidates() -> List[str]:
    patterns = [
        os.path.join(BASE_DIR, "Train", "**", "weights", "best.pt"),
        os.path.join(BASE_DIR, "Train", "**", "weights", "last.pt"),
        os.path.join(BASE_DIR, "*.pt"),
    ]
    candidates: List[str] = []
    seen: set[str] = set()

    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            full = os.path.abspath(path)
            if full in seen:
                continue
            if not os.path.isfile(full):
                continue
            seen.add(full)
            candidates.append(full)

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates


def _build_default_model_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    for index, model_path in enumerate(_discover_model_candidates()):
        rel_path = os.path.relpath(model_path, BASE_DIR).replace("\\", "/")
        rel_no_ext = os.path.splitext(rel_path)[0]
        model_id = _to_slug(rel_no_ext)
        if model_id in used_ids:
            model_id = f"{model_id}_{index + 1}"
        used_ids.add(model_id)

        entries.append(
            {
                "id": model_id,
                "name": os.path.basename(os.path.dirname(os.path.dirname(model_path))),
                "path": rel_path,
                "label": "tbm",
                "class_tag": "tbm",
                "enabled": True,
                "conf": 0.1,
                "imgsz": 640,
            }
        )

    return entries


def _normalize_model_entry(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    model_id_raw = str(entry.get("id", "")).strip()
    model_name_raw = str(entry.get("name", "")).strip()
    model_path_raw = str(entry.get("path", "")).strip()

    model_id = _to_slug(model_id_raw or model_name_raw or f"model_{index + 1}")
    model_name = model_name_raw or model_id
    class_tag_raw = str(entry.get("class_tag", "")).strip().lower()
    if class_tag_raw:
        class_tag = _normalize_class_tag(class_tag_raw)
    else:
        infer_text = (
            f"{model_name_raw} {model_path_raw} {entry.get('label', '')}".lower()
        )
        has_tbm = re.search(r"(^|[^a-z0-9])tbm([^a-z0-9]|$)", infer_text) is not None
        has_tm = re.search(r"(^|[^a-z0-9])tm([^a-z0-9]|$)", infer_text) is not None
        if has_tbm:
            class_tag = "tbm"
        elif has_tm:
            class_tag = "tm"
        else:
            class_tag = "tbm"
    label = str(entry.get("label", "")).strip()
    if not label or label == "palm_tree":
        label = class_tag

    try:
        conf = float(entry.get("conf", 0.1))
    except (TypeError, ValueError):
        conf = 0.1
    conf = min(1.0, max(0.0, conf))

    try:
        imgsz = int(entry.get("imgsz", 640))
    except (TypeError, ValueError):
        imgsz = 640
    imgsz = max(64, imgsz)

    return {
        "id": model_id,
        "name": model_name,
        "path": model_path_raw,
        "label": label,
        "class_tag": class_tag,
        "enabled": bool(entry.get("enabled", True)),
        "conf": conf,
        "imgsz": imgsz,
    }


def _normalize_model_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        item = _normalize_model_entry(entry, index)
        model_id = item["id"]
        suffix = 2
        while model_id in used_ids:
            model_id = f"{item['id']}_{suffix}"
            suffix += 1
        item["id"] = model_id
        used_ids.add(model_id)
        normalized.append(item)

    return normalized


def _load_model_config_entries() -> List[Dict[str, Any]]:
    if not os.path.exists(MODEL_CONFIG_PATH):
        default_entries = _normalize_model_entries(_build_default_model_entries())
        _save_model_config_entries(default_entries)
        return default_entries

    try:
        with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except Exception as exc:
        print(f"[WARN] Failed to read model config: {exc}")
        raw = None

    if isinstance(raw, dict):
        raw_entries = raw.get("models", [])
    elif isinstance(raw, list):
        raw_entries = raw
    else:
        raw_entries = []

    entries = _normalize_model_entries(raw_entries)
    if not entries:
        entries = _normalize_model_entries(_build_default_model_entries())
        _save_model_config_entries(entries)
    return entries


def _save_model_config_entries(entries: List[Dict[str, Any]]) -> None:
    payload = {"models": entries}
    with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _entries_for_detector(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        item["path"] = _resolve_model_path(item.get("path", ""))
        prepared.append(item)
    return prepared


MODEL_CONFIG_ENTRIES = _load_model_config_entries()

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
os.makedirs(SAVE_DIR, exist_ok=True)

# Global Detector
detector = MultiModelDetector(_entries_for_detector(MODEL_CONFIG_ENTRIES))


def _parse_model_ids(model_ids_raw: Optional[str]) -> Optional[List[str]]:
    if not model_ids_raw:
        return None

    value = model_ids_raw.strip()
    if not value:
        return None

    parsed_ids: List[str] = []
    if value.startswith("["):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                parsed_ids = [str(item).strip() for item in loaded if str(item).strip()]
        except json.JSONDecodeError:
            parsed_ids = []

    if not parsed_ids:
        parsed_ids = [item.strip() for item in value.split(",") if item.strip()]

    return parsed_ids or None


def _parse_zoom_scales(zoom_scales_raw: Optional[str]) -> Optional[List[float]]:
    if not zoom_scales_raw:
        return None

    value = zoom_scales_raw.strip()
    if not value:
        return None

    parsed_scales: List[float] = []
    if value.startswith("["):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                for item in loaded:
                    try:
                        scale = float(item)
                    except (TypeError, ValueError):
                        continue
                    if scale > 0:
                        parsed_scales.append(scale)
        except json.JSONDecodeError:
            parsed_scales = []

    if not parsed_scales:
        for item in value.split(","):
            try:
                scale = float(item.strip())
            except (TypeError, ValueError):
                continue
            if scale > 0:
                parsed_scales.append(scale)

    return parsed_scales or None


def _build_models_response_payload() -> Dict[str, Any]:
    loaded_ids = set(detector.get_loaded_model_ids())
    errors = detector.get_model_errors()
    models: List[Dict[str, Any]] = []

    for entry in MODEL_CONFIG_ENTRIES:
        model_id = entry.get("id", "")
        resolved_path = _resolve_model_path(entry.get("path", ""))
        class_tag = _normalize_class_tag(entry.get("class_tag", "tbm"))
        modified_at, modified_ts = _format_model_mtime(resolved_path)
        model_payload = {
            "id": model_id,
            "name": entry.get("name", model_id),
            "label": entry.get("label", "palm_tree"),
            "class_tag": class_tag,
            "model_tag": class_tag.upper(),
            "path": entry.get("path", ""),
            "resolved_path": resolved_path,
            "enabled": bool(entry.get("enabled", True)),
            "conf": float(entry.get("conf", 0.1)),
            "imgsz": int(entry.get("imgsz", 640)),
            "exists": os.path.isfile(resolved_path),
            "loaded": model_id in loaded_ids,
            "modified_at": modified_at,
            "modified_ts": modified_ts,
            "error": errors.get(model_id),
        }
        models.append(model_payload)

    auto_best_by_class: Dict[str, Optional[str]] = {
        k: None for k in SUPPORTED_INFER_CLASSES
    }
    for class_tag in SUPPORTED_INFER_CLASSES:
        class_candidates = [
            item
            for item in models
            if str(item.get("class_tag", "tbm")) == class_tag
            and bool(item.get("enabled"))
            and bool(item.get("loaded"))
            and bool(item.get("exists"))
        ]
        class_candidates.sort(
            key=lambda item: float(item.get("modified_ts") or 0.0), reverse=True
        )
        if class_candidates:
            auto_best_by_class[class_tag] = (
                str(class_candidates[0].get("id", "")) or None
            )

    return {
        "models": models,
        "loaded_model_ids": sorted(list(loaded_ids)),
        "errors": errors,
        "config_path": MODEL_CONFIG_PATH,
        "auto_best_by_class": auto_best_by_class,
    }


def _pick_effective_model_ids(
    selected_model_ids: Optional[List[str]],
    infer_class: str = "tbm",
) -> List[str]:
    infer_class_norm = _normalize_class_tag(infer_class)
    payload = _build_models_response_payload()
    models: List[Dict[str, Any]] = payload.get("models", [])
    by_id = {str(item.get("id", "")): item for item in models}

    # If manual selection is provided, we still filter by class and take the first valid one
    # to avoid running multiple models for the same class unless explicitly intended.
    # User request: "inference ini hanya menjalankan dua class sjaa, misal clas tbm dan tm"
    if selected_model_ids:
        for model_id in selected_model_ids:
            item = by_id.get(model_id)
            if not item:
                continue
            if str(item.get("class_tag", "tbm")) != infer_class_norm:
                continue
            if not bool(item.get("enabled")) or not bool(item.get("loaded")):
                continue
            if not bool(item.get("exists")):
                continue
            return [model_id]

    # Automatic selection: newest modified model in requested class.
    class_candidates = [
        item
        for item in models
        if str(item.get("class_tag", "tbm")) == infer_class_norm
        and bool(item.get("enabled"))
        and bool(item.get("loaded"))
        and bool(item.get("exists"))
    ]

    if not class_candidates:
        return []

    # Sort by modification time (newest first)
    class_candidates.sort(
        key=lambda item: float(item.get("modified_ts") or 0.0), reverse=True
    )

    return [str(class_candidates[0].get("id", ""))]


@app.get("/api/models")
async def get_models():
    return _build_models_response_payload()


@app.post("/api/models/reload")
async def reload_models():
    global MODEL_CONFIG_ENTRIES
    MODEL_CONFIG_ENTRIES = _load_model_config_entries()
    detector.reload(_entries_for_detector(MODEL_CONFIG_ENTRIES))
    return {
        "status": "success",
        **_build_models_response_payload(),
    }


@app.get("/api/models/config")
async def get_model_config():
    return {
        "path": MODEL_CONFIG_PATH,
        "models": MODEL_CONFIG_ENTRIES,
    }


@app.post("/api/models/config")
async def set_model_config(payload: Any = Body(...)):
    global MODEL_CONFIG_ENTRIES

    if isinstance(payload, dict):
        raw_models = payload.get("models", [])
    elif isinstance(payload, list):
        raw_models = payload
    else:
        raise HTTPException(
            status_code=400, detail="Invalid payload. Use {'models': [...]}."
        )

    models = _normalize_model_entries(raw_models)
    if not models:
        raise HTTPException(status_code=400, detail="Model list cannot be empty")

    MODEL_CONFIG_ENTRIES = models
    _save_model_config_entries(MODEL_CONFIG_ENTRIES)
    detector.reload(_entries_for_detector(MODEL_CONFIG_ENTRIES))

    return {
        "status": "success",
        **_build_models_response_payload(),
    }


@app.get("/api/models/validate")
async def validate_models(path: Optional[str] = None):
    if path:
        return TreeDetector.validate_model_path(_resolve_model_path(path))

    results: List[Dict[str, Any]] = []
    for entry in MODEL_CONFIG_ENTRIES:
        model_path = str(entry.get("path", "")).strip()
        if not model_path:
            results.append(
                {
                    "id": entry.get("id", ""),
                    "name": entry.get("name", entry.get("id", "")),
                    "path": "",
                    "exists": False,
                    "valid": False,
                    "error": "Missing model path",
                }
            )
            continue

        validation = TreeDetector.validate_model_path(_resolve_model_path(model_path))
        validation["id"] = entry.get("id", "")
        validation["name"] = entry.get("name", entry.get("id", ""))
        results.append(validation)

    return {"results": results}


@app.get("/api/images")
async def list_images():
    """List TIFF images by auto-scanning configured directories."""
    return discover_images()


def get_full_path(provided_path: str):
    """Utility to resolve paths safely - only allows discovered images."""
    discovered = discover_images()
    discovered_paths = {img["path"] for img in discovered}
    debug_log("PATH", f"resolve request={provided_path}")
    if os.path.isabs(provided_path):
        if provided_path in discovered_paths and os.path.exists(provided_path):
            debug_log("PATH", f"resolved ok={provided_path}")
            return provided_path
        debug_log("PATH", f"forbidden path={provided_path}")
        raise HTTPException(status_code=403, detail="Path not allowed")
    debug_log("PATH", f"not found path={provided_path}")
    raise HTTPException(status_code=404, detail="Image not found")


def _resolve_target_dir(output_dir: str = None):
    target_dir = output_dir.strip() if isinstance(output_dir, str) else ""
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        return target_dir
    os.makedirs(DEFAULT_EXPORT_DIR, exist_ok=True)
    return DEFAULT_EXPORT_DIR


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
            "overviews": src.overviews(1),
        }


@app.get("/api/image-info")
async def get_image_info(path: str):
    try:
        full_path = get_full_path(path)
        info = get_cached_image_info(full_path)
        debug_log(
            "IMAGE",
            f"image-info ok path={full_path} size={info['width']}x{info['height']} bands={info['bands']}",
        )
        return info
    except HTTPException:
        debug_log("IMAGE", f"image-info http-error path={path}")
        raise
    except Exception as e:
        debug_log("IMAGE", f"image-info failed path={path} error={e}")
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
            resampling=rasterio.enums.Resampling.bilinear
            if scale < 1.0
            else rasterio.enums.Resampling.nearest,
        )

        if num_bands == 1:
            data = np.repeat(data, 3, axis=0)
        elif num_bands == 2:
            data = np.concatenate(
                [data, np.zeros((1, out_h, out_w), dtype=data.dtype)], axis=0
            )

        data = np.transpose(data, (1, 2, 0))

        if data.dtype != np.uint8:
            # Smart normalization: use a fixed range or dynamic based on sample
            data = (
                (data - data.min()) / (data.max() - data.min() + 1e-5) * 255
            ).astype(np.uint8)

        img = Image.fromarray(data)
        img_buffer = BytesIO()
        # Use WebP for better compression/speed
        img.save(img_buffer, format="WebP", quality=80)
        return Response(content=img_buffer.getvalue(), media_type="image/webp")


@app.post("/api/detect")
async def run_detection(
    path: str,
    x: int,
    y: int,
    w: int,
    h: int,
    conf: float = 0.01,
    imgsz: int = 640,
    model_ids: Optional[str] = Query(default=None),
    infer_class: str = Query(default="tbm"),
):
    """Run detection on a specific window."""
    full_path = get_full_path(path)
    selected_model_ids = _parse_model_ids(model_ids)
    effective_model_ids = _pick_effective_model_ids(
        selected_model_ids, infer_class=infer_class
    )
    if not effective_model_ids:
        raise HTTPException(
            status_code=400,
            detail=f"No active model available for class '{_normalize_class_tag(infer_class)}'",
        )
    print(
        f"[DETECT] path={path}, x={x}, y={y}, w={w}, h={h}, conf={conf}, imgsz={imgsz}, infer_class={_normalize_class_tag(infer_class)}, model_ids={effective_model_ids}"
    )
    try:
        points = detector.predict_tiff_slice(
            full_path,
            x,
            y,
            w,
            h,
            conf=conf,
            imgsz=imgsz,
            selected_model_ids=effective_model_ids,
        )
        print(f"[DETECT] Found {len(points)} points")
        return {"points": points}
    except Exception as e:
        print(f"[DETECT] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect-full")
async def run_full_detection(
    path: str,
    conf: float = 0.1,
    tile_size: int = 640,
    overlap: float = 0.25,
    window: str = None,
    polygon: str = None,
    imgsz: int = 640,
    model_ids: Optional[str] = Query(default=None),
    zoom_scales: Optional[str] = Query(default=None),
    infer_class: str = Query(default="tbm"),
    dbscan_eps: float = 12.0,
    cluster_method: str = "hybrid",
    max_passes: int = 2,
    min_new_points: int = 2,
    batch_size: int = 8,
):
    """
    Streaming SAHI-style full image or ROI detection (box or polygon).
    window: JSON string of [x, y, w, h]
    polygon: JSON string of [[x1, y1], [x2, y2], ...]
    tile_size: Scanning window size (default 640)
    imgsz: YOLO inference size (default 640, should match training size)
    infer_class: target class tag (tbm|tm), default tbm.
    dbscan_eps: DBSCAN epsilon for filtering nearby detections (default 12.0).
    cluster_method: none|grid|dbscan|hybrid (default hybrid).
    max_passes: max staggered overlap passes to avoid endless looping.
    min_new_points: early-stop threshold across passes.
    batch_size: number of tiles inferred per model call.
    """
    full_path = get_full_path(path)

    scan_window = None
    if window:
        try:
            scan_window = json.loads(window)
        except:
            pass

    scan_polygon = None
    if polygon:
        try:
            scan_polygon = json.loads(polygon)
        except:
            pass

    # If overlap is > 1, assume it's pixels. If < 1, assume percentage.
    actual_overlap = overlap
    if overlap >= 1:
        actual_overlap = overlap / tile_size

    selected_model_ids = _parse_model_ids(model_ids)
    parsed_zoom_scales = _parse_zoom_scales(zoom_scales)
    effective_model_ids = _pick_effective_model_ids(
        selected_model_ids, infer_class=infer_class
    )
    if not effective_model_ids:
        raise HTTPException(
            status_code=400,
            detail=f"No active model available for class '{_normalize_class_tag(infer_class)}'",
        )

    coverage_stats = None
    if scan_window or scan_polygon:
        try:
            debug_detector = TreeDetector.__new__(TreeDetector)
            debug_detector.model_id = "coverage_debug"
            debug_detector.label = "palm_tree"
            coverage_stats = TreeDetector.get_tile_coverage_stats(
                debug_detector,
                full_path,
                tile_size=tile_size,
                overlap=actual_overlap,
                scan_window=scan_window,
                polygon=scan_polygon,
                max_passes=max_passes,
            )
        except Exception as coverage_exc:
            print(f"[DETECT-COVERAGE] Failed to compute coverage stats: {coverage_exc}")

    print(
        f"[DETECT-STREAM] path={path}, window={scan_window}, poly={len(scan_polygon) if scan_polygon else 0}, tile_size={tile_size}, imgsz={imgsz}, infer_class={_normalize_class_tag(infer_class)}, model_ids={effective_model_ids}, zoom_scales={parsed_zoom_scales}, cluster_method={cluster_method}, max_passes={max_passes}, min_new_points={min_new_points}, batch_size={batch_size}"
    )
    if coverage_stats is not None:
        print(
            "[DETECT-COVERAGE] "
            f"ratio={coverage_stats['coverage_ratio']:.4f} "
            f"covered={coverage_stats['covered_pixels']}/{coverage_stats['roi_pixels']} "
            f"uncovered={coverage_stats['uncovered_pixels']} "
            f"safe_ratio={coverage_stats.get('safe_coverage_ratio', coverage_stats['coverage_ratio']):.4f} "
            f"safe_uncovered={coverage_stats.get('safe_uncovered_pixels', coverage_stats['uncovered_pixels'])} "
            f"min={coverage_stats['min_coverage']} max={coverage_stats['max_coverage']}"
        )

    async def event_generator():
        points_events = 0
        streamed_points = 0
        try:
            if coverage_stats is not None:
                debug_log("DETECT", f"emit coverage {coverage_stats}")
                yield f"data: {json.dumps({'type': 'coverage', 'coverage': coverage_stats})}\n\n"
            for update in detector.predict_tiff_full(
                full_path,
                conf=conf,
                tile_size=tile_size,
                overlap=actual_overlap,
                scan_window=scan_window,
                polygon=scan_polygon,
                imgsz=imgsz,
                selected_model_ids=effective_model_ids,
                dbscan_eps=dbscan_eps,
                cluster_method=cluster_method,
                max_passes=max_passes,
                min_new_points=min_new_points,
                batch_size=batch_size,
                zoom_scales=parsed_zoom_scales,
            ):
                update_type = str(update.get("type", "unknown"))
                if update_type == "points":
                    batch_count = len(update.get("points", []))
                    points_events += 1
                    streamed_points += batch_count
                    debug_log(
                        "DETECT",
                        f"emit points event={points_events} batch={batch_count} streamed_total={streamed_points} model={update.get('model_id')} pass={update.get('pass')}",
                    )
                elif update_type == "final":
                    debug_log(
                        "DETECT",
                        f"emit final total_found={len(update.get('points', []))} model={update.get('model_id')}",
                    )
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            print(f"[STREAM-ERROR] {e}")
            debug_log("DETECT", f"stream failed error={e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
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
    """Load points from a local JSON file if it exists."""
    fname = path.replace(os.sep, "_").replace("/", "_") + ".json"
    save_path = os.path.join(SAVE_DIR, fname)
    if not os.path.exists(save_path):
        return []

    with open(save_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/detect-full-coverage")
async def get_detection_coverage_map(
    path: str,
    tile_size: int = 640,
    overlap: float = 0.5,
    window: str = None,
    polygon: str = None,
    max_passes: int = 3,
    stats_only: bool = False,
):
    """Generate a debug visualization showing tile coverage of the detection area."""
    try:
        from PIL import Image
        import cv2

        full_path = get_full_path(path)
        scan_window = None
        if window:
            try:
                scan_window = json.loads(window)
            except:
                pass

        scan_polygon = None
        if polygon:
            try:
                scan_polygon = json.loads(polygon)
            except:
                pass

        detector_instance = TreeDetector.__new__(TreeDetector)
        detector_instance.model_id = "debug"
        detector_instance.label = "palm_tree"

        coverage_stats = TreeDetector.get_tile_coverage_stats(
            detector_instance,
            full_path,
            tile_size=tile_size,
            overlap=overlap,
            scan_window=scan_window,
            polygon=scan_polygon,
            max_passes=max_passes,
        )

        if stats_only:
            return coverage_stats

        coverage = TreeDetector.get_tile_coverage_map(
            detector_instance,
            full_path,
            tile_size=tile_size,
            overlap=overlap,
            scan_window=scan_window,
            polygon=scan_polygon,
            max_passes=max_passes,
        )

        if coverage.size == 0:
            return {"error": "No coverage data generated"}

        coverage_normalized = cv2.normalize(coverage, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(coverage_normalized, cv2.COLORMAP_JET)

        if scan_polygon and len(scan_polygon) >= 3:
            window_info = coverage_stats.get("scan_window") or [0, 0, coverage.shape[1], coverage.shape[0]]
            start_x = int(window_info[0])
            start_y = int(window_info[1])
            poly_arr = np.array(
                [[int(round(x - start_x)), int(round(y - start_y))] for x, y in scan_polygon],
                dtype=np.int32,
            )
            cv2.polylines(heatmap, [poly_arr], isClosed=True, color=(0, 255, 0), thickness=2)

        _, img_encoded = cv2.imencode('.png', heatmap)
        return StreamingResponse(
            BytesIO(img_encoded.tobytes()),
            media_type="image/png",
        )
    except Exception as e:
        print(f"[COVERAGE-ERROR] {e}")
        return {"error": str(e)}


@app.get("/api/export-dir")
async def get_export_dir():
    """Get current export directory."""
    return {"export_dir": DEFAULT_EXPORT_DIR}


@app.post("/api/export-dir")
async def set_export_dir(export_dir: str = Body(...)):
    """Set export directory for shapefile and dataset exports."""
    global DEFAULT_EXPORT_DIR
    if os.path.exists(export_dir) or os.path.isabs(export_dir):
        DEFAULT_EXPORT_DIR = export_dir
        os.makedirs(DEFAULT_EXPORT_DIR, exist_ok=True)
        return {"status": "success", "export_dir": DEFAULT_EXPORT_DIR}
    raise HTTPException(status_code=400, detail="Invalid directory path")


@app.get("/api/export-dir/pick")
async def pick_export_dir():
    """Open a native folder picker on the server machine and set export dir."""
    global DEFAULT_EXPORT_DIR
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        initial_dir = (
            DEFAULT_EXPORT_DIR if os.path.isdir(DEFAULT_EXPORT_DIR) else os.getcwd()
        )
        selected_dir = filedialog.askdirectory(
            initialdir=initial_dir,
            title="Select export directory",
            mustexist=False,
        )
        root.destroy()

        if not selected_dir:
            return {"status": "cancelled", "export_dir": DEFAULT_EXPORT_DIR}

        DEFAULT_EXPORT_DIR = selected_dir
        os.makedirs(DEFAULT_EXPORT_DIR, exist_ok=True)
        return {"status": "success", "export_dir": DEFAULT_EXPORT_DIR}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to open directory picker: {e}",
        )


@app.post("/api/export/shapefile")
async def export_shapefile(path: str, labels: list = Body(...), output_dir: str = None):
    """Export to Shapefile in user-specified directory and return metadata."""
    full_path = get_full_path(path)
    base_name = os.path.basename(path).split(".")[0]
    import time

    output_name = f"{base_name}_labels_{int(time.time())}.shp"
    target_dir = _resolve_target_dir(output_dir)
    output_path = os.path.join(target_dir, output_name)

    try:
        Exporter.to_shapefile(labels, full_path, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shapefile export failed: {e}")

    return {
        "status": "success",
        "path": output_path,
        "directory": target_dir,
        "filename": output_name,
        "base_name": base_name,
    }


@app.post("/api/export/shapefile/download")
async def export_shapefile_download(
    background_tasks: BackgroundTasks,
    path: str,
    labels: list = Body(...),
    output_dir: str = None,
):
    """Export shapefile and return a downloadable ZIP for remote clients."""
    full_path = get_full_path(path)
    base_name = os.path.basename(path).split(".")[0]
    import time

    ts = int(time.time())
    shp_base = f"{base_name}_labels_{ts}"
    target_dir = _resolve_target_dir(output_dir)
    shp_path = os.path.join(target_dir, f"{shp_base}.shp")

    try:
        Exporter.to_shapefile(labels, full_path, shp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shapefile export failed: {e}")

    temp_zip_path = os.path.join(tempfile.gettempdir(), f"{shp_base}.zip")
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg"]

    with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ext in exts:
            part = shp_path.replace(".shp", ext)
            if os.path.exists(part):
                zf.write(part, arcname=os.path.basename(part))

    def _cleanup_temp(p):
        if os.path.exists(p):
            os.remove(p)

    background_tasks.add_task(_cleanup_temp, temp_zip_path)

    return FileResponse(
        temp_zip_path,
        media_type="application/zip",
        filename=f"{shp_base}.zip",
    )


@app.post("/api/export/dataset")
async def export_dataset(path: str, labels: list = Body(...), output_dir: str = None):
    """Export to YOLO dataset in user-specified directory."""
    full_path = get_full_path(path)
    target_dir = _resolve_target_dir(output_dir)
    output_dir_path = os.path.join(
        target_dir, "dataset_" + os.path.basename(path).split(".")[0]
    )
    Exporter.to_yolo_dataset(labels, full_path, output_dir_path)
    return {"status": "success", "directory": output_dir_path}


@app.get("/api/import/shapefile")
async def import_shapefile(shp_path: str, ref_image: str = None):
    """
    Import a shapefile and convert geo coordinates to pixel coordinates
    using the reference image's georeferencing.
    Returns list of points with pixel coordinates.
    """
    import geopandas as gpd

    if not os.path.exists(shp_path):
        raise HTTPException(status_code=404, detail=f"Shapefile not found: {shp_path}")

    try:
        gdf = gpd.read_file(shp_path)
        print(f"[IMPORT] Loaded {len(gdf)} features from {shp_path}")

        points = []

        # If ref_image provided, use its transform/CRS.
        ref_transform = None
        ref_crs = None
        ref_width = None
        ref_height = None
        ref_image_full = None
        if ref_image:
            try:
                ref_image_full = get_full_path(ref_image)
            except Exception:
                ref_image_full = ref_image

        if ref_image_full and os.path.exists(ref_image_full):
            with rasterio.open(ref_image_full) as src:
                ref_transform = src.transform
                ref_crs = src.crs
                ref_width = src.width
                ref_height = src.height

        if gdf.crs:
            shp_crs = gdf.crs
            print(f"[IMPORT] Shapefile CRS: {shp_crs}")
        else:
            shp_crs = None

        if ref_crs is not None and shp_crs is not None and shp_crs != ref_crs:
            try:
                gdf = gdf.to_crs(ref_crs)
                print(f"[IMPORT] Reprojected shapefile CRS to: {ref_crs}")
            except Exception as reproj_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to reproject shapefile CRS to reference image CRS: {reproj_err}",
                )

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue

            if geom.geom_type == "Point":
                x, y = geom.x, geom.y
            elif geom.geom_type == "MultiPoint":
                x, y = geom.centroid.x, geom.centroid.y
            else:
                # For polygon/line geometries, use centroid to allow import.
                try:
                    c = geom.centroid
                    x, y = c.x, c.y
                except Exception:
                    continue

            # Convert to pixel coordinates if we have reference transform
            if ref_transform is not None:
                # Use floor here because exported shapefile points are pixel centers.
                # Inverse affine of a center point returns n + 0.5; round() causes +/-1 drift.
                col, row_idx = ~ref_transform * (x, y)
                px, py = int(np.floor(col)), int(np.floor(row_idx))
            else:
                px, py = int(x), int(y)

            if ref_width is not None and ref_height is not None:
                if px < 0 or py < 0 or px >= ref_width or py >= ref_height:
                    continue

            points.append(
                {
                    "x": px,
                    "y": py,
                    "label": str(row.get("label", "imported")),
                    "conf": float(row.get("conf", 1.0)),
                }
            )

        print(f"[IMPORT] Converted {len(points)} points to pixel coordinates")
        return {"status": "success", "points": points, "count": len(points)}

    except Exception as e:
        print(f"[IMPORT-ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files for React (if built)
_ACTIVE_STATIC_DIR = get_active_static_dir()
if _ACTIVE_STATIC_DIR:
    _ASSETS_DIR = os.path.join(_ACTIVE_STATIC_DIR, "assets")
    if os.path.isdir(_ASSETS_DIR):
        app.mount(
            "/assets",
            StaticFiles(directory=_ASSETS_DIR),
            name="static-assets",
        )

    for _filename in ("favicon.svg", "icons.svg"):
        _file_path = os.path.join(_ACTIVE_STATIC_DIR, _filename)
        if os.path.isfile(_file_path):
            async def _serve_static_file(filename: str = _filename, file_path: str = _file_path):
                return FileResponse(file_path, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})

            app.add_api_route(f"/{_filename}", _serve_static_file, methods=["GET"])

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend_index():
        debug_log("FRONTEND", f"serve index dir={_ACTIVE_STATIC_DIR}")
        return HTMLResponse(
            content=_build_index_html(_ACTIVE_STATIC_DIR),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    async def serve_frontend_app(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("assets/"):
            raise HTTPException(status_code=404, detail="Not found")

        candidate_path = os.path.join(_ACTIVE_STATIC_DIR, full_path)
        if os.path.isfile(candidate_path):
            debug_log("FRONTEND", f"serve file path={candidate_path}")
            return FileResponse(
                candidate_path,
                headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
            )

        debug_log("FRONTEND", f"spa fallback path={full_path} dir={_ACTIVE_STATIC_DIR}")
        return HTMLResponse(
            content=_build_index_html(_ACTIVE_STATIC_DIR),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
