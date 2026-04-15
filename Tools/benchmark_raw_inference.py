"""Benchmark raw-raster inference against shapefile ground truth points.

This script evaluates full-raster detection settings on the original TIFF +
shapefile pairs before tiling. It ranks model/parameter combinations using a
balanced score across recall, precision, F1, and counting accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import rasterio
from shapely.geometry import MultiPoint, Point, Polygon, MultiPolygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_APP_DIR = PROJECT_ROOT / "auto_labeling" / "python_app"
if str(PYTHON_APP_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_APP_DIR))

from core.detector import MultiModelDetector  # noqa: E402


RAW_DATASETS = [
    {
        "name": "Atha_Kundo",
        "raster": PROJECT_ROOT
        / "Dataset"
        / "Raw Dataset"
        / "TBM"
        / "Processed"
        / "Clipped"
        / "Dataset_1A_Atha.tif",
        "shapefile": PROJECT_ROOT
        / "Dataset"
        / "Raw Dataset"
        / "TBM"
        / "TBM_1A_Kundo.shp",
    },
    {
        "name": "Isnin_TBM",
        "raster": PROJECT_ROOT
        / "Dataset"
        / "Raw Dataset"
        / "TBM"
        / "Processed"
        / "Clipped"
        / "tbm_1a_isnin.tif",
        "shapefile": PROJECT_ROOT
        / "Dataset"
        / "Raw Dataset"
        / "TBM"
        / "TBM"
        / "TBM.shp",
    },
]

DEFAULT_MODELS = [
    PROJECT_ROOT / "Train" / "runs" / "ft_yolo11s_640_v2" / "weights" / "best.pt",
    PROJECT_ROOT
    / "Train"
    / "runs"
    / "train_13_4_2026_tbm_only"
    / "weights"
    / "best.pt",
    PROJECT_ROOT / "Train" / "runs" / "exp2_yolo11s_1280" / "weights" / "best.pt",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    conf: float
    tile_size: int
    overlap: float
    imgsz: int
    zoom_scales: Tuple[float, ...]
    cluster_method: str
    dbscan_eps: float
    max_passes: int = 2
    min_new_points: int = 2
    batch_size: int = 8

    def config_id(self) -> str:
        zoom_text = "-".join(_format_zoom_scale(value) for value in self.zoom_scales)
        return (
            f"conf{self.conf:g}_tile{self.tile_size}_ov{self.overlap:g}_img{self.imgsz}_"
            f"zoom{zoom_text}_{self.cluster_method}_eps{self.dbscan_eps:g}"
        )

    def to_detector_kwargs(self) -> Dict[str, Any]:
        return {
            "conf": self.conf,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "imgsz": self.imgsz,
            "dbscan_eps": self.dbscan_eps,
            "cluster_method": self.cluster_method,
            "max_passes": self.max_passes,
            "min_new_points": self.min_new_points,
            "batch_size": self.batch_size,
            "zoom_scales": list(self.zoom_scales),
        }


def _format_zoom_scale(value: float) -> str:
    rounded = round(float(value), 4)
    if rounded.is_integer():
        return str(int(rounded))
    return str(rounded).replace(".", "p")


def _slugify_path(path: Path) -> str:
    text = path.stem.lower()
    return (
        "".join(char if char.isalnum() else "_" for char in text).strip("_") or "model"
    )


def _parse_float_list(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values or None


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    values = _parse_float_list(raw)
    if values is None:
        return None
    return [int(value) for value in values]


def _parse_zoom_sets(raw: Optional[str]) -> Optional[List[Tuple[float, ...]]]:
    if not raw:
        return None
    sets: List[Tuple[float, ...]] = []
    for group in raw.split(";"):
        group = group.strip()
        if not group:
            continue
        scales = tuple(float(item.strip()) for item in group.split(",") if item.strip())
        if scales:
            sets.append(scales)
    return sets or None


def _build_model_entries(model_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for index, model_path in enumerate(model_paths, start=1):
        entries.append(
            {
                "id": f"model_{index}_{_slugify_path(model_path)}",
                "name": model_path.parents[1].name,
                "path": str(model_path.resolve()),
                "label": "tbm",
                "class_tag": "tbm",
                "enabled": True,
                "conf": 0.01,
                "imgsz": 640,
            }
        )
    return entries


def _extract_geometry_points(geometry: Any) -> Iterable[Tuple[float, float]]:
    if geometry is None:
        return []
    if isinstance(geometry, Point):
        return [(geometry.x, geometry.y)]
    if isinstance(geometry, MultiPoint):
        return [
            (point.x, point.y) for point in geometry.geoms if isinstance(point, Point)
        ]
    if isinstance(geometry, (Polygon, MultiPolygon)):
        centroid = geometry.centroid
        return [(centroid.x, centroid.y)]
    centroid = geometry.centroid
    return [(centroid.x, centroid.y)]


def load_ground_truth_points(
    raster_path: Path, shapefile_path: Path
) -> List[Tuple[float, float]]:
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    points: List[Tuple[float, float]] = []
    for geometry in gdf.geometry:
        for gx, gy in _extract_geometry_points(geometry):
            px, py = ~transform * (gx, gy)
            if 0 <= px < width and 0 <= py < height:
                points.append((float(px), float(py)))

    return points


def greedy_match_points(
    gt_points: Sequence[Tuple[float, float]],
    pred_points: Sequence[Dict[str, Any]],
    tolerance_px: float,
) -> Dict[str, Any]:
    matched_gt: set[int] = set()
    matched_pairs: List[Dict[str, Any]] = []
    false_positives: List[Dict[str, Any]] = []

    ordered_predictions = sorted(
        pred_points,
        key=lambda item: float(item.get("conf", 0.0)),
        reverse=True,
    )

    for prediction in ordered_predictions:
        px = float(prediction["x"])
        py = float(prediction["y"])
        best_index = None
        best_distance = tolerance_px

        for gt_index, (gx, gy) in enumerate(gt_points):
            if gt_index in matched_gt:
                continue
            distance = math.hypot(px - gx, py - gy)
            if distance <= best_distance:
                best_index = gt_index
                best_distance = distance

        if best_index is None:
            false_positives.append(prediction)
            continue

        matched_gt.add(best_index)
        matched_pairs.append(
            {
                "gt_index": best_index,
                "pred_x": px,
                "pred_y": py,
                "gt_x": gt_points[best_index][0],
                "gt_y": gt_points[best_index][1],
                "distance_px": best_distance,
                "conf": float(prediction.get("conf", 0.0)),
            }
        )

    false_negatives = [
        {"gt_index": index, "x": gx, "y": gy}
        for index, (gx, gy) in enumerate(gt_points)
        if index not in matched_gt
    ]

    tp = len(matched_pairs)
    fp = len(false_positives)
    fn = len(false_negatives)
    recall = tp / len(gt_points) if gt_points else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    gt_count = len(gt_points)
    pred_count = len(pred_points)
    count_error = abs(pred_count - gt_count)
    count_accuracy = max(0.0, 1.0 - (count_error / gt_count)) if gt_count else 0.0
    mean_distance = (
        sum(item["distance_px"] for item in matched_pairs) / tp if tp else None
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "count_error": count_error,
        "count_accuracy": count_accuracy,
        "mean_match_distance_px": mean_distance,
    }


def run_detector(
    detector: MultiModelDetector,
    raster_path: Path,
    model_id: str,
    config: BenchmarkConfig,
) -> Tuple[List[Dict[str, Any]], float]:
    start = time.time()
    final_points: List[Dict[str, Any]] = []

    for update in detector.predict_tiff_full(
        tiff_path=str(raster_path),
        selected_model_ids=[model_id],
        **config.to_detector_kwargs(),
    ):
        if update.get("type") == "final":
            final_points = list(update.get("points", []))

    return final_points, time.time() - start


def build_benchmark_configs(args: argparse.Namespace) -> List[BenchmarkConfig]:
    if args.quick:
        return [
            BenchmarkConfig(0.01, 640, 0.25, 960, (1.0, 1.5, 2.0), "hybrid", 10.0),
            BenchmarkConfig(0.02, 640, 0.25, 960, (1.0, 1.5, 2.0), "hybrid", 12.0),
            BenchmarkConfig(0.01, 896, 0.25, 1280, (1.0, 1.5), "hybrid", 12.0),
            BenchmarkConfig(0.01, 640, 0.35, 960, (1.0, 1.5, 2.0), "dbscan", 10.0),
        ]

    confs = _parse_float_list(args.confs) or [0.01]
    tile_sizes = _parse_int_list(args.tile_sizes) or [640]
    overlaps = _parse_float_list(args.overlaps) or [0.25]
    imgszs = _parse_int_list(args.imgszs) or [960]
    zoom_sets = _parse_zoom_sets(args.zoom_sets) or [(1.0, 1.5, 2.0)]
    methods = [
        item.strip()
        for item in (args.cluster_methods or "hybrid").split(",")
        if item.strip()
    ]
    dbscan_eps_values = _parse_float_list(args.dbscan_eps_values) or [10.0]
    max_passes_values = _parse_int_list(args.max_passes_values) or [2]
    min_new_points_values = _parse_int_list(args.min_new_points_values) or [2]
    batch_sizes = _parse_int_list(args.batch_sizes) or [8]

    configs: List[BenchmarkConfig] = []
    for values in product(
        confs,
        tile_sizes,
        overlaps,
        imgszs,
        zoom_sets,
        methods,
        dbscan_eps_values,
        max_passes_values,
        min_new_points_values,
        batch_sizes,
    ):
        configs.append(BenchmarkConfig(*values))
    return configs


def aggregate_dataset_metrics(dataset_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    recalls = [item["recall"] for item in dataset_metrics]
    precisions = [item["precision"] for item in dataset_metrics]
    f1_scores = [item["f1"] for item in dataset_metrics]
    count_accuracies = [item["count_accuracy"] for item in dataset_metrics]
    runtimes = [item["runtime_sec"] for item in dataset_metrics]

    total_tp = sum(item["tp"] for item in dataset_metrics)
    total_fp = sum(item["fp"] for item in dataset_metrics)
    total_fn = sum(item["fn"] for item in dataset_metrics)
    total_gt = sum(item["gt_count"] for item in dataset_metrics)
    total_pred = sum(item["pred_count"] for item in dataset_metrics)
    total_count_error = sum(item["count_error"] for item in dataset_metrics)

    macro_recall = sum(recalls) / len(recalls)
    macro_precision = sum(precisions) / len(precisions)
    macro_f1 = sum(f1_scores) / len(f1_scores)
    macro_count_accuracy = sum(count_accuracies) / len(count_accuracies)
    mean_runtime = sum(runtimes) / len(runtimes)

    micro_recall = total_tp / total_gt if total_gt else 0.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_f1 = (
        2 * micro_recall * micro_precision / (micro_recall + micro_precision)
        if (micro_recall + micro_precision)
        else 0.0
    )
    micro_count_accuracy = (
        max(0.0, 1.0 - (total_count_error / total_gt)) if total_gt else 0.0
    )

    score = (
        0.30 * macro_recall
        + 0.20 * macro_precision
        + 0.25 * macro_f1
        + 0.25 * macro_count_accuracy
    )

    return {
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "macro_count_accuracy": macro_count_accuracy,
        "micro_recall": micro_recall,
        "micro_precision": micro_precision,
        "micro_f1": micro_f1,
        "micro_count_accuracy": micro_count_accuracy,
        "mean_runtime_sec": mean_runtime,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_count_error": total_count_error,
        "score": score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark raw raster inference settings"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run a small curated sweep"
    )
    parser.add_argument(
        "--models", type=str, default=None, help="Comma-separated .pt model paths"
    )
    parser.add_argument(
        "--dataset-names",
        type=str,
        default=None,
        help="Comma-separated dataset names to benchmark",
    )
    parser.add_argument(
        "--confs", type=str, default=None, help="Comma-separated confidence values"
    )
    parser.add_argument(
        "--tile-sizes", type=str, default=None, help="Comma-separated tile sizes"
    )
    parser.add_argument(
        "--overlaps", type=str, default=None, help="Comma-separated overlap values"
    )
    parser.add_argument(
        "--imgszs", type=str, default=None, help="Comma-separated imgsz values"
    )
    parser.add_argument(
        "--zoom-sets",
        type=str,
        default=None,
        help="Semicolon-separated zoom groups, e.g. 1;1,1.5;1,1.5,2",
    )
    parser.add_argument(
        "--cluster-methods",
        type=str,
        default="hybrid",
        help="Comma-separated clustering methods",
    )
    parser.add_argument(
        "--dbscan-eps-values",
        type=str,
        default=None,
        help="Comma-separated dbscan eps values",
    )
    parser.add_argument(
        "--max-passes-values",
        type=str,
        default=None,
        help="Comma-separated max_passes values",
    )
    parser.add_argument(
        "--min-new-points-values",
        type=str,
        default=None,
        help="Comma-separated min_new_points values",
    )
    parser.add_argument(
        "--batch-sizes", type=str, default=None, help="Comma-separated batch sizes"
    )
    parser.add_argument(
        "--tolerance-px", type=float, default=12.0, help="GT matching radius in pixels"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "Train" / "runs" / "raw_benchmark"),
        help="Directory for benchmark outputs",
    )
    return parser.parse_args()


def write_outputs(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    csv_path = output_dir / "benchmark_summary.csv"
    if results:
        fieldnames = [
            "rank",
            "model_name",
            "model_path",
            "config_id",
            "score",
            "macro_recall",
            "macro_precision",
            "macro_f1",
            "macro_count_accuracy",
            "micro_recall",
            "micro_precision",
            "micro_f1",
            "micro_count_accuracy",
            "total_gt",
            "total_pred",
            "total_count_error",
            "mean_runtime_sec",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, result in enumerate(results, start=1):
                aggregate = result["aggregate"]
                writer.writerow(
                    {
                        "rank": index,
                        "model_name": result["model_name"],
                        "model_path": result["model_path"],
                        "config_id": result["config"]["config_id"],
                        "score": f"{aggregate['score']:.6f}",
                        "macro_recall": f"{aggregate['macro_recall']:.6f}",
                        "macro_precision": f"{aggregate['macro_precision']:.6f}",
                        "macro_f1": f"{aggregate['macro_f1']:.6f}",
                        "macro_count_accuracy": f"{aggregate['macro_count_accuracy']:.6f}",
                        "micro_recall": f"{aggregate['micro_recall']:.6f}",
                        "micro_precision": f"{aggregate['micro_precision']:.6f}",
                        "micro_f1": f"{aggregate['micro_f1']:.6f}",
                        "micro_count_accuracy": f"{aggregate['micro_count_accuracy']:.6f}",
                        "total_gt": aggregate["total_gt"],
                        "total_pred": aggregate["total_pred"],
                        "total_count_error": aggregate["total_count_error"],
                        "mean_runtime_sec": f"{aggregate['mean_runtime_sec']:.4f}",
                    }
                )


def persist_results(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    sorted_results = sorted(
        results, key=lambda item: item["aggregate"]["score"], reverse=True
    )
    write_outputs(output_dir, sorted_results)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    dataset_name_filter = None
    if args.dataset_names:
        dataset_name_filter = {
            item.strip() for item in args.dataset_names.split(",") if item.strip()
        }

    datasets = [
        item
        for item in RAW_DATASETS
        if dataset_name_filter is None or item["name"] in dataset_name_filter
    ]
    if not datasets:
        raise SystemExit("No matching datasets selected for benchmarking")

    gt_lookup: Dict[str, List[Tuple[float, float]]] = {}
    print("Loading ground-truth points from shapefiles...")
    for dataset in datasets:
        raster_path = Path(dataset["raster"])
        shapefile_path = Path(dataset["shapefile"])
        if not raster_path.exists():
            raise SystemExit(f"Raster not found: {raster_path}")
        if not shapefile_path.exists():
            raise SystemExit(f"Shapefile not found: {shapefile_path}")
        gt_points = load_ground_truth_points(raster_path, shapefile_path)
        gt_lookup[dataset["name"]] = gt_points
        print(f"  {dataset['name']}: {len(gt_points)} GT points")

    model_paths = (
        [Path(item.strip()) for item in args.models.split(",") if item.strip()]
        if args.models
        else DEFAULT_MODELS
    )
    model_paths = [path for path in model_paths if path.exists()]
    if not model_paths:
        raise SystemExit("No valid model weights found for benchmarking")

    detector = MultiModelDetector(_build_model_entries(model_paths))
    loaded_model_ids = detector.get_loaded_model_ids()
    if not loaded_model_ids:
        raise SystemExit(
            f"Detector failed to load models: {detector.get_model_errors()}"
        )

    model_id_to_entry = {entry["id"]: entry for entry in detector.model_entries}
    benchmark_configs = build_benchmark_configs(args)
    print(f"Loaded {len(loaded_model_ids)} model(s)")
    print(f"Prepared {len(benchmark_configs)} benchmark configuration(s)")

    all_results: List[Dict[str, Any]] = []
    total_runs = len(loaded_model_ids) * len(benchmark_configs) * len(datasets)
    run_index = 0

    for model_id in loaded_model_ids:
        entry = model_id_to_entry[model_id]
        model_path = Path(entry["path"])
        print(f"\n=== Model: {entry['name']} ({model_path}) ===")

        for config in benchmark_configs:
            print(f"\nConfig: {config.config_id()}")
            dataset_metrics: List[Dict[str, Any]] = []
            for dataset in datasets:
                run_index += 1
                print(
                    f"  [{run_index}/{total_runs}] Dataset {dataset['name']} ...",
                    end=" ",
                    flush=True,
                )
                predictions, runtime_sec = run_detector(
                    detector=detector,
                    raster_path=Path(dataset["raster"]),
                    model_id=model_id,
                    config=config,
                )
                metrics = greedy_match_points(
                    gt_points=gt_lookup[dataset["name"]],
                    pred_points=predictions,
                    tolerance_px=args.tolerance_px,
                )
                metrics["runtime_sec"] = runtime_sec
                metrics["dataset_name"] = dataset["name"]
                dataset_metrics.append(metrics)
                print(
                    f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
                    f"R={metrics['recall']:.4f} P={metrics['precision']:.4f} "
                    f"F1={metrics['f1']:.4f} CountErr={metrics['count_error']} "
                    f"Time={runtime_sec:.1f}s"
                )

            aggregate = aggregate_dataset_metrics(dataset_metrics)
            result = {
                "model_id": model_id,
                "model_name": entry["name"],
                "model_path": str(model_path),
                "config": {**asdict(config), "config_id": config.config_id()},
                "aggregate": aggregate,
                "datasets": dataset_metrics,
            }
            all_results.append(result)
            persist_results(output_dir, all_results)
            print(
                "  Aggregate: "
                f"score={aggregate['score']:.4f} "
                f"macro_recall={aggregate['macro_recall']:.4f} "
                f"macro_precision={aggregate['macro_precision']:.4f} "
                f"macro_f1={aggregate['macro_f1']:.4f} "
                f"macro_count_accuracy={aggregate['macro_count_accuracy']:.4f}"
            )

    all_results.sort(key=lambda item: item["aggregate"]["score"], reverse=True)
    write_outputs(output_dir, all_results)

    print("\n=== Top Results ===")
    for index, result in enumerate(all_results[:5], start=1):
        aggregate = result["aggregate"]
        print(
            f"{index}. {result['model_name']} | {result['config']['config_id']} | "
            f"score={aggregate['score']:.4f} | "
            f"R={aggregate['macro_recall']:.4f} P={aggregate['macro_precision']:.4f} "
            f"F1={aggregate['macro_f1']:.4f} CountAcc={aggregate['macro_count_accuracy']:.4f}"
        )

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
