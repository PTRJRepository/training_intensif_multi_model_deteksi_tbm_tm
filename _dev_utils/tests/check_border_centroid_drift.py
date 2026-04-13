"""Small scripted check for centroid drift near tile borders.

Compares:
- OLD merge proxy: unweighted centroid (equal weight)
- NEW merge: confidence * edge_weight weighted centroid

Runs on a small ROI for speed and reports before/after drift against a
cluster reference point (highest confidence*edge_weight sample).
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from typing import Dict, List, Tuple

import numpy as np
import rasterio


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_APP = os.path.join(REPO_ROOT, "auto_labeling", "python_app")
if PY_APP not in sys.path:
    sys.path.insert(0, PY_APP)

from core.detector import TreeDetector  # noqa: E402


def collect_raw_points(
    detector: TreeDetector,
    tiff_path: str,
    conf: float,
    imgsz: int,
    tile_size: int,
    overlap: float,
    roi: Tuple[int, int, int, int],
) -> List[Dict[str, float]]:
    sx, sy, sw, sh = roi
    overlap = max(0.0, min(0.9, overlap))
    stride = max(10, tile_size - int(tile_size * overlap))
    edge_band = max(8.0, tile_size * overlap * 0.5)

    with rasterio.open(tiff_path) as src:
        x_starts = sorted(
            set(
                list(range(sx, max(sx + 1, sx + sw - tile_size), stride))
                + [max(sx, sx + sw - tile_size)]
            )
        )
        y_starts = sorted(
            set(
                list(range(sy, max(sy + 1, sy + sh - tile_size), stride))
                + [max(sy, sy + sh - tile_size)]
            )
        )

        all_points: List[Dict[str, float]] = []
        tile_index = 0
        for y_pos in y_starts:
            for x_pos in x_starts:
                tile_index += 1
                w = min(tile_size, sx + sw - x_pos)
                h = min(tile_size, sy + sh - y_pos)
                if w < 10 or h < 10:
                    continue

                window = rasterio.windows.Window(x_pos, y_pos, w, h)
                num_bands = min(src.count, 3)
                data = src.read(list(range(1, num_bands + 1)), window=window)

                if num_bands == 1:
                    data = np.repeat(data, 3, axis=0)
                elif num_bands == 2:
                    data = np.concatenate(
                        [data, np.zeros((1, h, w), dtype=data.dtype)], axis=0
                    )

                data = np.transpose(data, (1, 2, 0))
                if data.dtype != np.uint8:
                    sample = data[::4, ::4]
                    p2, p98 = np.percentile(sample, (2, 98))
                    if p98 > p2:
                        data = np.clip(data, p2, p98)
                        data = ((data - p2) / (p98 - p2) * 255).astype(np.uint8)
                    else:
                        data = data.astype(np.uint8)

                input_data = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                input_data[0:h, 0:w, :] = data
                if float(input_data.mean()) <= 5.0:
                    continue

                results = detector.model.predict(
                    source=input_data,
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                    augment=False,
                )

                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for i, box in enumerate(boxes):
                        left = float(max(0.0, min(box[0], w)))
                        right = float(max(0.0, min(box[2], w)))
                        top = float(max(0.0, min(box[1], h)))
                        bottom = float(max(0.0, min(box[3], h)))
                        if right <= left or bottom <= top:
                            continue

                        cx_local = (left + right) / 2.0
                        cy_local = (top + bottom) / 2.0
                        edge_margin = min(
                            cx_local,
                            cy_local,
                            max(0.0, w - cx_local),
                            max(0.0, h - cy_local),
                        )
                        edge_weight = max(0.2, min(1.0, edge_margin / edge_band))

                        all_points.append(
                            {
                                "x": cx_local + x_pos,
                                "y": cy_local + y_pos,
                                "conf": float(confs[i]),
                                "edge_weight": float(edge_weight),
                                "tile_id": float(tile_index),
                            }
                        )

    return all_points


def cluster_points(
    points: List[Dict[str, float]], dist_threshold: float
) -> List[List[int]]:
    if not points:
        return []
    cell_size = float(dist_threshold)
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, p in enumerate(points):
        gx = int(p["x"] // cell_size)
        gy = int(p["y"] // cell_size)
        grid.setdefault((gx, gy), []).append(idx)

    visited = [False] * len(points)
    clusters: List[List[int]] = []

    for i in range(len(points)):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        current: List[int] = []

        while queue:
            cur = queue.pop()
            current.append(cur)
            cx = points[cur]["x"]
            cy = points[cur]["y"]
            gx = int(cx // cell_size)
            gy = int(cy // cell_size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for nxt in grid.get((gx + dx, gy + dy), []):
                        if visited[nxt]:
                            continue
                        nx = points[nxt]["x"]
                        ny = points[nxt]["y"]
                        if math.hypot(cx - nx, cy - ny) <= dist_threshold:
                            visited[nxt] = True
                            queue.append(nxt)
        clusters.append(current)

    return clusters


def centroid_unweighted(cluster: List[Dict[str, float]]) -> Tuple[float, float]:
    return (
        sum(p["x"] for p in cluster) / len(cluster),
        sum(p["y"] for p in cluster) / len(cluster),
    )


def centroid_weighted(cluster: List[Dict[str, float]]) -> Tuple[float, float]:
    total = 0.0
    x_acc = 0.0
    y_acc = 0.0
    for p in cluster:
        w = max(1e-6, float(p["conf"]) * float(p["edge_weight"]))
        total += w
        x_acc += p["x"] * w
        y_acc += p["y"] * w
    return (x_acc / total, y_acc / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=os.path.join(
            REPO_ROOT, "Train", "runs", "train_13_4_2026_tbm_only", "weights", "best.pt"
        ),
    )
    parser.add_argument(
        "--image",
        default=os.path.join(
            REPO_ROOT, "Dataset", "Processed", "Clipped", "1_clipped.tif"
        ),
    )
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--roi", default="0,0,1920,1920", help="x,y,w,h")
    parser.add_argument("--dist-threshold", type=float, default=12.0)
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    roi = tuple(int(v.strip()) for v in args.roi.split(","))
    if len(roi) != 4:
        raise ValueError("--roi must be x,y,w,h")

    detector = TreeDetector(args.weights)
    print(f"[INFO] Model  : {args.weights}")
    print(f"[INFO] Image  : {args.image}")
    print(
        f"[INFO] Params : tile={args.tile_size}, overlap={args.overlap}, imgsz={args.imgsz}, conf={args.conf}, roi={roi}"
    )

    raw_points = collect_raw_points(
        detector=detector,
        tiff_path=args.image,
        conf=args.conf,
        imgsz=args.imgsz,
        tile_size=args.tile_size,
        overlap=args.overlap,
        roi=roi,
    )
    print(f"[INFO] Raw detections collected: {len(raw_points)}")

    clusters_idx = cluster_points(raw_points, args.dist_threshold)
    print(f"[INFO] Clusters formed: {len(clusters_idx)}")

    old_errors: List[float] = []
    new_errors: List[float] = []
    old_new_shift: List[float] = []
    checked_clusters = 0

    examples: List[Tuple[float, float, float, int, int]] = []

    for idxs in clusters_idx:
        if len(idxs) < 2:
            continue
        cluster = [raw_points[i] for i in idxs]
        tiles = {int(p["tile_id"]) for p in cluster}
        if len(tiles) < 2:
            continue

        # Focus on border-sensitive cases.
        if min(p["edge_weight"] for p in cluster) > 0.6:
            continue

        checked_clusters += 1
        old_c = centroid_unweighted(cluster)
        new_c = centroid_weighted(cluster)

        ref = max(cluster, key=lambda p: float(p["conf"]) * float(p["edge_weight"]))
        ref_xy = (ref["x"], ref["y"])

        old_err = math.hypot(old_c[0] - ref_xy[0], old_c[1] - ref_xy[1])
        new_err = math.hypot(new_c[0] - ref_xy[0], new_c[1] - ref_xy[1])
        shift = math.hypot(old_c[0] - new_c[0], old_c[1] - new_c[1])

        old_errors.append(old_err)
        new_errors.append(new_err)
        old_new_shift.append(shift)
        examples.append((old_err, new_err, shift, len(cluster), len(tiles)))

    print(f"[INFO] Border-sensitive multi-tile clusters checked: {checked_clusters}")
    if checked_clusters == 0:
        print(
            "[WARN] No suitable border-sensitive clusters found. Try larger ROI or lower conf."
        )
        return

    def _safe_mean(vals: List[float]) -> float:
        return float(statistics.mean(vals)) if vals else 0.0

    def _safe_median(vals: List[float]) -> float:
        return float(statistics.median(vals)) if vals else 0.0

    print("\n=== Drift Summary (pixels) ===")
    print(f"OLD  mean error vs ref: {_safe_mean(old_errors):.3f}")
    print(f"NEW  mean error vs ref: {_safe_mean(new_errors):.3f}")
    print(f"OLD median error vs ref: {_safe_median(old_errors):.3f}")
    print(f"NEW median error vs ref: {_safe_median(new_errors):.3f}")
    print(f"Mean centroid shift OLD->NEW: {_safe_mean(old_new_shift):.3f}")

    mean_old = _safe_mean(old_errors)
    mean_new = _safe_mean(new_errors)
    improvement = ((mean_old - mean_new) / mean_old * 100.0) if mean_old > 0 else 0.0
    print(f"Relative improvement (mean): {improvement:.2f}%")

    examples.sort(key=lambda t: t[2], reverse=True)
    print(
        "\nTop 5 centroid shifts (old_err, new_err, shift, cluster_size, tile_count):"
    )
    for row in examples[:5]:
        print("  ", tuple(round(v, 3) if isinstance(v, float) else v for v in row))


if __name__ == "__main__":
    main()
