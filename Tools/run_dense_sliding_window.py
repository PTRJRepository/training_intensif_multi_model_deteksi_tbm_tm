"""
Dense Sliding Window Inference with Multiple Overlap Passes

Script ini menjalankan deteksi dengan banyak konfigurasi sliding window sekaligus:
- Beberapa tile_size (320, 480, 640) untuk menangkap objek kecil dan besar
- Overlap tinggi (0.5, 0.65, 0.75) agar tidak ada area yang terlewat
- Multi-scale zoom (1.0, 1.5, 2.0) untuk detail lebih halus
- Multi-pass staggered untuk cover boundary tiles
- Grid size calculation yang tepat agar seluruh ROI tercakup

Hasil dari semua pass digabung dan di-cluster untuk menghasilkan output final.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directories to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "auto_labeling" / "python_app"))

import rasterio
from rasterio.windows import Window
import numpy as np
from shapely.geometry import Polygon

from core.detector import TreeDetector, MultiModelDetector


def compute_grid_coverage(
    roi_width: int,
    roi_height: int,
    tile_size: int,
    overlap: float,
) -> Dict[str, int]:
    """Hitung jumlah tile yang dibutuhkan untuk cover seluruh ROI tanpa gap."""
    stride = max(10, tile_size - int(tile_size * overlap))
    cols = math.ceil(max(0, roi_width - tile_size) / stride) + 1 if roi_width >= tile_size else 1
    rows = math.ceil(max(0, roi_height - tile_size) / stride) + 1 if roi_height >= tile_size else 1
    total_tiles = cols * rows
    # Hitung area yang benar-benar tercover
    covered_w = tile_size + (cols - 1) * stride
    covered_h = tile_size + (rows - 1) * stride
    actual_coverage = min(covered_w, roi_width) * min(covered_h, roi_height)
    roi_area = roi_width * roi_height
    coverage_pct = (actual_coverage / roi_area * 100) if roi_area > 0 else 0

    return {
        "cols": cols,
        "rows": rows,
        "total_tiles": total_tiles,
        "stride": stride,
        "coverage_pct": round(coverage_pct, 2),
        "overlap_pct": round(overlap * 100, 1),
    }


def generate_dense_configs(
    tile_sizes: List[int] = None,
    overlaps: List[float] = None,
    zoom_scales: List[float] = None,
    conf: float = 0.1,
    imgsz: int = 640,
) -> List[Dict]:
    """Generate semua kombinasi sliding window config."""
    tile_sizes = tile_sizes or [320, 480, 640]
    overlaps = overlaps or [0.5, 0.65, 0.75]
    zoom_scales = zoom_scales or [1.0, 1.5, 2.0]

    configs = []
    for ts in tile_sizes:
        for ov in overlaps:
            # imgsz minimal = scaled tile size terbesar
            max_scaled = int(ts * max(zoom_scales))
            effective_imgsz = max(imgsz, max_scaled)

            grid_info = compute_grid_coverage(
                roi_width=10000,  # placeholder, akan dihitung ulang per image
                roi_height=10000,
                tile_size=ts,
                overlap=ov,
            )

            configs.append({
                "tile_size": ts,
                "overlap": ov,
                "imgsz": effective_imgsz,
                "zoom_scales": zoom_scales,
                "conf": conf,
                "max_passes": 3,
                "min_new_points": 1,  # Lebih sensitif
                "grid_info_template": grid_info,
            })

    return configs


def run_dense_inference(
    tiff_path: str,
    model_path: str,
    scan_window: Optional[List[float]] = None,
    polygon: Optional[List[List[float]]] = None,
    configs: Optional[List[Dict]] = None,
    cluster_method: str = "hybrid",
    dbscan_eps: float = 8.0,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Jalankan dense sliding window inference dengan banyak config sekaligus.

    Returns dict dengan points, stats, dan grid coverage info.
    """
    # Default configs jika tidak disediakan
    if configs is None:
        configs = generate_dense_configs(
            tile_sizes=[320, 480, 640],
            overlaps=[0.5, 0.65, 0.75],
            zoom_scales=[1.0, 1.5, 2.0],
            conf=0.1,
        )

    # Load model
    detector = TreeDetector(
        model_path=model_path,
        conf=0.01,  # Sangat rendah, filtering nanti di post-process
        device=0,
    )

    # Get ROI dimensions
    with rasterio.open(tiff_path) as src:
        img_width, img_height = src.width, src.height

        if scan_window:
            sx, sy, sw, sh = scan_window
            start_x = max(0, int(sx))
            start_y = max(0, int(sy))
            end_x = min(img_width, int(sx + sw))
            end_y = min(img_height, int(sy + sh))
        else:
            start_x, start_y = 0, 0
            end_x, end_y = img_width, img_height

        roi_w = end_x - start_x
        roi_h = end_y - start_y

    print(f"\n{'='*60}")
    print(f"ROI: {roi_w} x {roi_h} pixels ({roi_w * roi_h / 1e6:.1f} MP)")
    print(f"{'='*60}\n")

    # Print grid coverage untuk setiap config
    print("Grid Coverage Analysis:")
    print(f"{'Tile Size':<12} {'Overlap':<10} {'Stride':<10} {'Tiles':<10} {'Coverage':<10}")
    print(f"{'-'*52}")

    all_grid_infos = []
    for cfg in configs:
        grid_info = compute_grid_coverage(roi_w, roi_h, cfg["tile_size"], cfg["overlap"])
        all_grid_infos.append(grid_info)
        print(
            f"{cfg['tile_size']:<12} "
            f"{grid_info['overlap_pct']:<10} "
            f"{grid_info['stride']:<10} "
            f"{grid_info['total_tiles']:<10} "
            f"{grid_info['coverage_pct']:<10}%"
        )

    print(f"\n{'='*60}")
    print(f"Running {len(configs)} sliding window configurations...")
    print(f"{'='*60}\n")

    # Jalankan semua configs
    all_points: List[Dict] = []
    total_tiles_processed = 0

    for i, cfg in enumerate(configs):
        ts = cfg["tile_size"]
        ov = cfg["overlap"]
        imgsz = cfg["imgsz"]
        zoom = cfg["zoom_scales"]
        conf = cfg["conf"]

        print(f"\n[{i+1}/{len(configs)}] tile={ts}, overlap={ov*100:.0f}%, imgsz={imgsz}, zoom={zoom}")
        t_start = time.time()

        points_this_config = []
        for update in detector.predict_tiff_full(
            tiff_path=tiff_path,
            conf=conf,
            tile_size=ts,
            overlap=ov,
            scan_window=scan_window,
            polygon=polygon,
            imgsz=imgsz,
            cluster_method="none",  # Jangan cluster dulu, nanti di akhir
            max_passes=cfg["max_passes"],
            min_new_points=cfg["min_new_points"],
            zoom_scales=zoom,
            batch_size=16,
        ):
            if update["type"] == "window":
                total_tiles_processed += 1
            elif update["type"] == "points":
                points_this_config = update["points"]

        elapsed = time.time() - t_start
        n_points = len(points_this_config)
        all_points.extend(points_this_config)

        print(f"  -> {n_points} raw detections in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"Total raw points (sebelum clustering): {len(all_points)}")
    print(f"Total tiles processed: {total_tiles_processed}")
    print(f"{'='*60}\n")

    # Clustering final - gunakan DBSCAN dengan eps lebih kecil untuk akurasi tinggi
    print(f"Clustering dengan {cluster_method} (eps={dbscan_eps})...")
    t_cluster = time.time()

    # Convert points ke format yang bisa di-cluster
    if all_points:
        final_points = detector.filter_nearby_points(
            all_points,
            method=cluster_method,
            dbscan_eps=dbscan_eps,
        )
    else:
        final_points = []

    cluster_time = time.time() - t_cluster
    print(f"Clustering selesai dalam {cluster_time:.2f}s -> {len(final_points)} points")

    # Simpan hasil
    result = {
        "points": final_points,
        "total_detections": len(final_points),
        "stats": {
            "total_raw_points": len(all_points),
            "total_tiles_processed": total_tiles_processed,
            "roi_dimensions": {"width": roi_w, "height": roi_h},
            "configs_run": len(configs),
            "cluster_method": cluster_method,
            "dbscan_eps": dbscan_eps,
            "clustering_time_sec": round(cluster_time, 2),
        },
        "grid_coverage": all_grid_infos,
        "configs_used": [
            {k: v for k, v in cfg.items() if k != "grid_info_template"}
            for cfg in configs
        ],
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Save points sebagai JSON
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nHasil disimpan ke: {out}")

        # Save juga sebagai shapefile jika ada rasterio CRS
        with rasterio.open(tiff_path) as src:
            if src.crs:
                from core.exporter import to_shapefile

                shp_path = str(out.with_suffix(".shp"))
                # Convert points ke format yang sesuai
                geo_points = []
                for p in final_points:
                    px, py = p["x"], p["y"]
                    # Convert pixel ke geocoordinate
                    geo_x, geo_y = src.xy(py, px)  # row, col -> x, y
                    geo_points.append({"x": geo_x, "y": geo_y, "conf": p.get("conf", 0)})

                to_shapefile(
                    points=geo_points,
                    output_path=shp_path,
                    crs=src.crs,
                )
                print(f"Shapefile disimpan ke: {shp_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Dense Sliding Window Inference")
    parser.add_argument("tiff", help="Path to TIFF file")
    parser.add_argument("--model", default=None, help="Path to YOLO .pt model")
    parser.add_argument("--scan-window", nargs=4, type=float, help="x y w h scan window")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument(
        "--tile-sizes",
        nargs="+",
        type=int,
        default=[320, 480, 640],
        help="Tile sizes to try",
    )
    parser.add_argument(
        "--overlaps",
        nargs="+",
        type=float,
        default=[0.5, 0.65, 0.75],
        help="Overlap ratios to try",
    )
    parser.add_argument(
        "--zoom-scales",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0],
        help="Zoom scales for multi-scale",
    )
    parser.add_argument(
        "--cluster-method",
        default="hybrid",
        choices=["hybrid", "dbscan", "grid", "none"],
    )
    parser.add_argument("--dbscan-eps", type=float, default=8.0, help="DBSCAN eps in pixels")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: auto)",
    )
    args = parser.parse_args()

    # Default model path
    if args.model is None:
        args.model = str(
            Path(__file__).parent.parent
            / "Train"
            / "runs"
            / "model_fine_tuning_15_04"
            / "weights"
            / "best.pt"
        )

    # Default output
    if args.output is None:
        tiff_name = Path(args.tiff).stem
        args.output = str(
            Path(__file__).parent.parent
            / "inference"
            / "dense_sliding_window"
            / f"{tiff_name}_dense_results.json"
        )

    scan_window = args.scan_window

    # Generate configs
    configs = generate_dense_configs(
        tile_sizes=args.tile_sizes,
        overlaps=args.overlaps,
        zoom_scales=args.zoom_scales,
        conf=args.conf,
    )

    print(f"\nModel: {args.model}")
    print(f"TIFF: {args.tiff}")
    print(f"Configs: {len(configs)} kombinasi sliding window")

    result = run_dense_inference(
        tiff_path=args.tiff,
        model_path=args.model,
        scan_window=scan_window,
        configs=configs,
        cluster_method=args.cluster_method,
        dbscan_eps=args.dbscan_eps,
        output_path=args.output,
    )

    print(f"\n{'='*60}")
    print(f"FINAL: {result['total_detections']} TBM terdeteksi")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
