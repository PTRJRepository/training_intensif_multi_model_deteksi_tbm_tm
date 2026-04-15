from __future__ import annotations

import math
import os
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN
from ultralytics import YOLO


class TreeDetector:
    MIN_BOX_SIZE_PX = 2.0
    MAX_BOX_SIZE_PX = 160.0
    MAX_BOX_ASPECT_RATIO = 2.75
    FAST_CLUSTER_THRESHOLD = 15000
    MIN_SAFE_CONTEXT_PX = 48

    def __init__(
        self,
        model_path: str,
        model_id: str = "default",
        label: str = "palm_tree",
        name: str = "Default Detector",
        default_conf: float = 0.1,
        default_imgsz: int = 640,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.model_id = model_id
        self.label = label
        self.name = name
        self.default_conf = float(default_conf)
        self.default_imgsz = int(default_imgsz)
        self.model = YOLO(model_path)
        self._predict_kwargs = self._build_predict_kwargs()

    @staticmethod
    def _build_predict_kwargs() -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            # Lower IoU suppresses duplicate micro-boxes earlier and reduces
            # NMS blow-ups on dense palm scenes.
            "iou": 0.45,
            # Keep a practical cap for dense scenes without overly slowing NMS.
            "max_det": 2200,
            "agnostic_nms": True,
        }
        try:
            import torch

            if torch.cuda.is_available():
                kwargs["half"] = True
        except Exception:
            # Fall back silently when torch is not available in this runtime.
            pass
        return kwargs

    @staticmethod
    def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
        if data.dtype == np.uint8:
            return data

        sample = data[::4, ::4]
        p2, p98 = np.percentile(sample, (2, 98))
        if p98 > p2:
            data = np.clip(data, p2, p98)
            return ((data - p2) / (p98 - p2) * 255).astype(np.uint8)
        return data.astype(np.uint8)

    @staticmethod
    def _to_three_channels(data: np.ndarray, h: int, w: int) -> np.ndarray:
        num_bands = data.shape[0]
        if num_bands == 1:
            data = np.repeat(data, 3, axis=0)
        elif num_bands == 2:
            data = np.concatenate([data, np.zeros((1, h, w), dtype=data.dtype)], axis=0)
        return np.transpose(data, (1, 2, 0))

    @staticmethod
    def _compute_axis_starts(
        start: int,
        end: int,
        tile_size: int,
        stride: int,
        offset: int = 0,
    ) -> List[int]:
        stride = max(1, int(stride))
        scan_span = end - start
        if scan_span <= tile_size:
            return [start]

        last_start = end - tile_size
        # Always keep the first and last anchors so each pass covers
        # ROI edges (top/left and bottom/right), even with staggered offsets.
        shifted_start = start + (max(0, int(offset)) % stride)
        shifted_start = max(start, min(shifted_start, last_start))
        starts = [start]
        starts.extend(range(shifted_start, last_start + 1, stride))
        starts.append(last_start)
        return sorted(set(starts))

    def _collect_tile_jobs(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        tile_size: int,
        stride: int,
        pass_offset: int,
        poly_obj: Optional[Polygon],
    ) -> List[Tuple[int, int, int, int]]:
        x_starts = self._compute_axis_starts(
            start_x, end_x, tile_size, stride, offset=pass_offset
        )
        y_starts = self._compute_axis_starts(
            start_y, end_y, tile_size, stride, offset=pass_offset
        )

        jobs: List[Tuple[int, int, int, int]] = []
        for y_pos in y_starts:
            for x_pos in x_starts:
                w = min(tile_size, end_x - x_pos)
                h = min(tile_size, end_y - y_pos)
                if w < 10 or h < 10:
                    continue

                if poly_obj:
                    tile_poly = Polygon(
                        [
                            (x_pos, y_pos),
                            (x_pos + w, y_pos),
                            (x_pos + w, y_pos + h),
                            (x_pos, y_pos + h),
                        ]
                    )
                    if not poly_obj.intersects(tile_poly):
                        continue

                jobs.append((x_pos, y_pos, w, h))

        return jobs

    @classmethod
    def _compute_safe_margin(cls, tile_size: int, overlap: float) -> int:
        overlap = max(0.0, float(overlap))
        return max(
            cls.MIN_SAFE_CONTEXT_PX,
            min(tile_size // 3, int(round(max(tile_size * overlap * 0.5, tile_size * 0.1)))),
        )

    @staticmethod
    def _build_roi_mask(
        shape: Tuple[int, int],
        start_x: int,
        start_y: int,
        poly_obj: Optional[Polygon],
    ) -> np.ndarray:
        if poly_obj is None:
            return np.ones(shape, dtype=bool)

        return rasterize(
            [(poly_obj, 1)],
            out_shape=shape,
            transform=rasterio.transform.Affine.translation(start_x, start_y),
            fill=0,
            default_value=1,
            all_touched=True,
            dtype=np.uint8,
        ).astype(bool)

    @staticmethod
    def _apply_jobs_to_coverage(
        coverage: np.ndarray,
        jobs: List[Tuple[int, int, int, int]],
        start_x: int,
        start_y: int,
        img_width: int,
        img_height: int,
        safe_margin: int = 0,
    ) -> None:
        for x_pos, y_pos, w, h in jobs:
            left = x_pos - start_x
            top = y_pos - start_y
            right = left + w
            bottom = top + h

            if safe_margin > 0:
                left_margin = 0 if x_pos <= 0 else safe_margin
                top_margin = 0 if y_pos <= 0 else safe_margin
                right_margin = 0 if x_pos + w >= img_width else safe_margin
                bottom_margin = 0 if y_pos + h >= img_height else safe_margin

                left += min(left_margin, max(0, w - 1))
                top += min(top_margin, max(0, h - 1))
                right -= min(right_margin, max(0, w - 1))
                bottom -= min(bottom_margin, max(0, h - 1))

                if right <= left or bottom <= top:
                    left = x_pos - start_x
                    top = y_pos - start_y
                    right = left + w
                    bottom = top + h

            coverage[top:bottom, left:right] += 1.0

    def _generate_gap_patch_jobs(
        self,
        gap_mask: np.ndarray,
        start_x: int,
        start_y: int,
        img_width: int,
        img_height: int,
        tile_size: int,
        poly_obj: Optional[Polygon],
        existing_jobs: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        if gap_mask.size == 0 or not np.any(gap_mask):
            return []

        cell_size = max(32, min(160, tile_size // 4))
        seen = set(existing_jobs)
        patch_jobs: List[Tuple[int, int, int, int]] = []
        max_x_start = max(0, img_width - tile_size)
        max_y_start = max(0, img_height - tile_size)
        mask_h, mask_w = gap_mask.shape

        for cell_y in range(0, mask_h, cell_size):
            for cell_x in range(0, mask_w, cell_size):
                cell = gap_mask[
                    cell_y:min(mask_h, cell_y + cell_size),
                    cell_x:min(mask_w, cell_x + cell_size),
                ]
                if not np.any(cell):
                    continue

                ys, xs = np.nonzero(cell)
                gx = start_x + cell_x + int(round(float(xs.mean())))
                gy = start_y + cell_y + int(round(float(ys.mean())))

                x_pos = max(0, min(max_x_start, int(round(gx - tile_size / 2))))
                y_pos = max(0, min(max_y_start, int(round(gy - tile_size / 2))))
                w = min(tile_size, img_width - x_pos)
                h = min(tile_size, img_height - y_pos)
                if w < 10 or h < 10:
                    continue

                if poly_obj is not None:
                    tile_poly = Polygon(
                        [
                            (x_pos, y_pos),
                            (x_pos + w, y_pos),
                            (x_pos + w, y_pos + h),
                            (x_pos, y_pos + h),
                        ]
                    )
                    if not poly_obj.intersects(tile_poly):
                        continue

                job = (x_pos, y_pos, w, h)
                if job in seen:
                    continue

                seen.add(job)
                patch_jobs.append(job)

        return patch_jobs

    def _plan_tile_passes(
        self,
        img_width: int,
        img_height: int,
        tile_size: int,
        overlap: float,
        scan_window: Optional[List[float]],
        poly_obj: Optional[Polygon],
        max_passes: int,
    ) -> Dict[str, Any]:
        effective_scan_window = scan_window
        if poly_obj is not None:
            minx, miny, maxx, maxy = poly_obj.bounds
            safe_margin = self._compute_safe_margin(tile_size, overlap)
            effective_scan_window = [
                minx - safe_margin,
                miny - safe_margin,
                (maxx - minx) + (safe_margin * 2),
                (maxy - miny) + (safe_margin * 2),
            ]
        else:
            safe_margin = self._compute_safe_margin(tile_size, overlap)

        if effective_scan_window:
            sx, sy, sw, sh = effective_scan_window
            start_x = max(0, math.floor(float(sx)))
            start_y = max(0, math.floor(float(sy)))
            end_x = min(img_width, math.ceil(float(sx + sw)))
            end_y = min(img_height, math.ceil(float(sy + sh)))
        else:
            start_x, start_y = 0, 0
            end_x, end_y = img_width, img_height

        stride = max(10, tile_size - int(tile_size * overlap))
        out_shape = (max(0, end_y - start_y), max(0, end_x - start_x))
        roi_mask = self._build_roi_mask(out_shape, start_x, start_y, poly_obj)
        gross_coverage = np.zeros(out_shape, dtype=np.float32)
        safe_coverage = np.zeros(out_shape, dtype=np.float32)

        pass_jobs: List[List[Tuple[int, int, int, int]]] = []
        safe_gap_pixels_by_pass: List[int] = []
        all_jobs: List[Tuple[int, int, int, int]] = []

        for pass_index in range(max_passes):
            pass_offset = int((stride * pass_index) / max(1, max_passes))
            jobs = self._collect_tile_jobs(
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                tile_size=tile_size,
                stride=stride,
                pass_offset=pass_offset,
                poly_obj=poly_obj,
            )
            if not jobs:
                continue

            pass_jobs.append(jobs)
            all_jobs.extend(jobs)
            self._apply_jobs_to_coverage(
                gross_coverage,
                jobs,
                start_x,
                start_y,
                img_width,
                img_height,
            )
            self._apply_jobs_to_coverage(
                safe_coverage,
                jobs,
                start_x,
                start_y,
                img_width,
                img_height,
                safe_margin=safe_margin,
            )
            safe_gap_pixels_by_pass.append(
                int(np.count_nonzero(np.logical_and(roi_mask, safe_coverage <= 0)))
            )

        patch_rounds = 0
        while (
            safe_gap_pixels_by_pass
            and safe_gap_pixels_by_pass[-1] > 0
            and patch_rounds < 3
        ):
            gap_mask = np.logical_and(roi_mask, safe_coverage <= 0)
            patch_jobs = self._generate_gap_patch_jobs(
                gap_mask=gap_mask,
                start_x=start_x,
                start_y=start_y,
                img_width=img_width,
                img_height=img_height,
                tile_size=tile_size,
                poly_obj=poly_obj,
                existing_jobs=all_jobs,
            )
            if not patch_jobs:
                break

            pass_jobs.append(patch_jobs)
            all_jobs.extend(patch_jobs)
            self._apply_jobs_to_coverage(
                gross_coverage,
                patch_jobs,
                start_x,
                start_y,
                img_width,
                img_height,
            )
            self._apply_jobs_to_coverage(
                safe_coverage,
                patch_jobs,
                start_x,
                start_y,
                img_width,
                img_height,
                safe_margin=safe_margin,
            )
            safe_gap_pixels_by_pass.append(
                int(np.count_nonzero(np.logical_and(roi_mask, safe_coverage <= 0)))
            )
            patch_rounds += 1

        return {
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "safe_margin": safe_margin,
            "pass_jobs": pass_jobs,
            "safe_gap_pixels_by_pass": safe_gap_pixels_by_pass,
            "gross_coverage": gross_coverage,
            "safe_coverage": safe_coverage,
            "roi_mask": roi_mask,
        }

    @staticmethod
    def _normalize_zoom_scales(zoom_scales: Optional[Iterable[float]]) -> List[float]:
        values: List[float] = []
        if zoom_scales is not None:
            for scale in zoom_scales:
                try:
                    value = float(scale)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    values.append(value)

        if not values:
            return [1.0]

        normalized = sorted({round(value, 4) for value in values})
        if 1.0 not in normalized:
            normalized.insert(0, 1.0)
        return normalized

    @staticmethod
    def _point_in_polygon(poly_obj: Polygon, x: float, y: float) -> bool:
        return bool(poly_obj.covers(Point(x, y)))

    @staticmethod
    def _should_process_tile(tile: np.ndarray) -> bool:
        return bool(tile.size and np.max(tile) > 0)

    @classmethod
    def _is_valid_detection_box(
        cls,
        left: float,
        top: float,
        right: float,
        bottom: float,
        scale_factor: float,
    ) -> bool:
        box_w = max(0.0, (right - left) / scale_factor)
        box_h = max(0.0, (bottom - top) / scale_factor)
        if box_w <= 0.0 or box_h <= 0.0:
            return False

        if box_w < cls.MIN_BOX_SIZE_PX or box_h < cls.MIN_BOX_SIZE_PX:
            return False
        if box_w > cls.MAX_BOX_SIZE_PX or box_h > cls.MAX_BOX_SIZE_PX:
            return False

        aspect_ratio = max(box_w / box_h, box_h / box_w)
        return aspect_ratio <= cls.MAX_BOX_ASPECT_RATIO

    def _predict_batched_jobs(
        self,
        batched_inputs: List[np.ndarray],
        batched_jobs: List[Tuple[int, int, int, int]],
        scale_factor: float,
        effective_conf: float,
        imgsz: int,
        edge_band: float,
        poly_obj: Optional[Polygon],
    ) -> List[Dict[str, Any]]:
        if not batched_inputs:
            return []

        results = self.model.predict(
            source=batched_inputs,
            conf=effective_conf,
            imgsz=imgsz,
            verbose=False,
            augment=False,
            **self._predict_kwargs,
        )

        points: List[Dict[str, Any]] = []
        scale_factor = max(0.0001, float(scale_factor))
        for job, result in zip(batched_jobs, results):
            x_pos, y_pos, w, h = job
            # The tile dimensions in SCALED pixel space (used for inference)
            sw = w * scale_factor
            sh = h * scale_factor

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for index, box in enumerate(boxes):
                # Use scaled dimensions for clipping, as YOLO coordinates match the input tile
                left = float(max(0.0, min(box[0], sw)))
                right = float(max(0.0, min(box[2], sw)))
                top = float(max(0.0, min(box[1], sh)))
                bottom = float(max(0.0, min(box[3], sh)))
                if right <= left or bottom <= top:
                    continue

                if not self._is_valid_detection_box(
                    left,
                    top,
                    right,
                    bottom,
                    scale_factor,
                ):
                    continue

                cx_local = (left + right) / 2.0
                cy_local = (top + bottom) / 2.0
                tx = (cx_local / scale_factor) + x_pos
                ty = (cy_local / scale_factor) + y_pos

                edge_margin = min(
                    cx_local,
                    cy_local,
                    max(0.0, sw - cx_local),
                    max(0.0, sh - cy_local),
                )
                edge_weight = max(0.2, min(1.0, edge_margin / (edge_band * scale_factor)))

                if poly_obj and not self._point_in_polygon(poly_obj, tx, ty):
                    continue

                points.append(
                    {
                        "x": float(tx),
                        "y": float(ty),
                        "label": self.label,
                        "conf": float(confs[index]),
                        "edge_weight": float(edge_weight),
                        "model_id": self.model_id,
                        "zoom_scale": float(scale_factor),
                    }
                )

        return points

    def predict_image(self, image_source: Any, conf: float = 0.01, imgsz: int = 640):
        results = self.model.predict(
            source=image_source, conf=conf, imgsz=imgsz, augment=True, verbose=False
        )
        points: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes.xywh.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for index, box in enumerate(boxes):
                points.append(
                    {
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "label": self.label,
                        "conf": float(confs[index]) if index < len(confs) else 1.0,
                        "model_id": self.model_id,
                    }
                )
        return points

    def predict_tiff_slice(
        self,
        tiff_path: str,
        x: int,
        y: int,
        w: int,
        h: int,
        conf: float = 0.01,
        imgsz: int = 640,
    ):
        with rasterio.open(tiff_path) as src:
            window = rasterio.windows.Window(x, y, w, h)
            num_bands = min(src.count, 3)
            data = src.read(list(range(1, num_bands + 1)), window=window)
            data = self._to_three_channels(data, h, w)
            data = self._normalize_to_uint8(data)

            results = self.model.predict(
                source=data, conf=conf, imgsz=imgsz, augment=True, verbose=False
            )

            points: List[Dict[str, Any]] = []
            for result in results:
                boxes = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for index, box in enumerate(boxes):
                    points.append(
                        {
                            "x": float(box[0] + x),
                            "y": float(box[1] + y),
                            "label": self.label,
                            "conf": float(confs[index]) if index < len(confs) else 1.0,
                            "model_id": self.model_id,
                        }
                    )
            return points

    def predict_tiff_full(
        self,
        tiff_path: str,
        conf: float = 0.1,
        tile_size: int = 640,
        overlap: float = 0.25,
        scan_window: Optional[List[float]] = None,
        polygon: Optional[List[List[float]]] = None,
        imgsz: int = 640,
        dbscan_eps: float = 12.0,
        cluster_method: str = "hybrid",
        max_passes: int = 2,
        min_new_points: int = 2,
        batch_size: int = 16,
        zoom_scales: Optional[Iterable[float]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        effective_conf = float(conf)
        max_passes = max(1, int(max_passes))
        min_new_points = max(0, int(min_new_points))
        batch_size = max(1, int(batch_size))
        cluster_method = str(cluster_method or "hybrid").strip().lower()
        zoom_scales_list = self._normalize_zoom_scales(zoom_scales)
        poly_obj: Optional[Polygon] = None

        if polygon and len(polygon) >= 3:
            poly_obj = Polygon(polygon)

        with rasterio.open(tiff_path) as src:
            img_width, img_height = src.width, src.height
            all_points_raw: List[Dict[str, Any]] = []
            emitted_raw_count = 0

            plan = self._plan_tile_passes(
                img_width=img_width,
                img_height=img_height,
                tile_size=tile_size,
                overlap=overlap,
                scan_window=scan_window,
                poly_obj=poly_obj,
                max_passes=max_passes,
            )
            start_x = int(plan["start_x"])
            start_y = int(plan["start_y"])
            end_x = int(plan["end_x"])
            end_y = int(plan["end_y"])
            pass_jobs = plan["pass_jobs"]
            safe_gap_pixels_by_pass = plan["safe_gap_pixels_by_pass"]

            scan_w = end_x - start_x
            scan_h = end_y - start_y
            if scan_w <= 0 or scan_h <= 0:
                yield {
                    "type": "final",
                    "points": [],
                    "model_id": self.model_id,
                    "total_found": 0,
                }
                return

            overlap = min(0.9, max(0.0, overlap))

            total_tiles = max(
                1, sum(len(jobs) for jobs in pass_jobs) * len(zoom_scales_list)
            )
            current_tile = 0
            best_filtered_count = 0

            # Optimize: Pre-calculate edge band for weighted centroid
            edge_band = max(8.0, tile_size * overlap * 0.5)

            for pass_index, jobs in enumerate(pass_jobs):
                for zoom_scale in zoom_scales_list:
                    scaled_tile_size = max(1, int(round(tile_size * zoom_scale)))
                    scaled_imgsz = max(int(imgsz), scaled_tile_size)
                    batched_inputs: List[np.ndarray] = []
                    batched_jobs: List[Tuple[int, int, int, int]] = []

                    def flush_batch() -> None:
                        nonlocal emitted_raw_count
                        if not batched_inputs:
                            return

                        batch_points = self._predict_batched_jobs(
                            batched_inputs=batched_inputs,
                            batched_jobs=batched_jobs,
                            scale_factor=zoom_scale,
                            effective_conf=effective_conf,
                            imgsz=scaled_imgsz,
                            edge_band=edge_band,
                            poly_obj=poly_obj,
                        )

                        if batch_points:
                            all_points_raw.extend(batch_points)
                            yield_points = all_points_raw[emitted_raw_count:]
                            emitted_raw_count = len(all_points_raw)
                            if yield_points:
                                yield {
                                    "type": "points",
                                    "points": yield_points,
                                    "model_id": self.model_id,
                                    "pass": pass_index + 1,
                                    "total_passes": len(pass_jobs),
                                    "zoom_scale": zoom_scale,
                                }

                        batched_inputs.clear()
                        batched_jobs.clear()

                    for x_pos, y_pos, w, h in jobs:
                        current_tile += 1
                        window = rasterio.windows.Window(x_pos, y_pos, w, h)
                        num_bands = min(src.count, 3)
                        scaled_w = max(1, int(round(w * zoom_scale)))
                        scaled_h = max(1, int(round(h * zoom_scale)))

                        data = src.read(
                            list(range(1, num_bands + 1)),
                            window=window,
                            out_shape=(num_bands, scaled_h, scaled_w),
                            resampling=(
                                rasterio.enums.Resampling.bilinear
                                if zoom_scale < 1.0
                                else rasterio.enums.Resampling.nearest
                            ),
                        )
                        data = self._to_three_channels(data, scaled_h, scaled_w)
                        data = self._normalize_to_uint8(data)

                        input_data = np.zeros(
                            (scaled_tile_size, scaled_tile_size, 3), dtype=np.uint8
                        )
                        input_data[0:scaled_h, 0:scaled_w, :] = data

                        yield {
                            "type": "window",
                            "window": [x_pos, y_pos, w, h],
                            "progress": current_tile / total_tiles,
                            "model_id": self.model_id,
                            "pass": pass_index + 1,
                            "total_passes": len(pass_jobs),
                            "zoom_scale": zoom_scale,
                        }

                        if not self._should_process_tile(input_data):
                            continue

                        batched_inputs.append(input_data)
                        batched_jobs.append((x_pos, y_pos, w, h))
                        if len(batched_inputs) >= batch_size:
                            yield from flush_batch()

                    yield from flush_batch()

                # Filter points after each pass to check for convergence
                filtered_points = self.filter_nearby_points(
                    all_points_raw,
                    eps=dbscan_eps,
                    method=cluster_method,
                )
                gained_points = len(filtered_points) - best_filtered_count
                best_filtered_count = len(filtered_points)
                safe_gap_pixels = (
                    safe_gap_pixels_by_pass[pass_index]
                    if pass_index < len(safe_gap_pixels_by_pass)
                    else 0
                )

                # Early stop if this pass didn't find significantly more objects
                if pass_index > 0 and gained_points <= min_new_points and safe_gap_pixels <= 0:
                    break

            # Final filtering and merging
            if all_points_raw:
                final_points = self.filter_nearby_points(
                    all_points_raw,
                    eps=dbscan_eps,
                    method=cluster_method,
                )
                yield {
                    "type": "final",
                    "points": final_points,
                    "model_id": self.model_id,
                    "total_found": len(final_points),
                }
            else:
                yield {
                    "type": "final",
                    "points": [],
                    "model_id": self.model_id,
                    "total_found": 0,
                }

    def filter_nearby_points(
        self,
        points: List[Dict[str, Any]],
        eps: float = 12.0,
        method: str = "hybrid",
        min_samples: int = 1,
    ) -> List[Dict[str, Any]]:
        eps = max(0.1, float(eps))
        normalized_method = str(method or "hybrid").strip().lower()
        if not points:
            return []

        if normalized_method == "none":
            return [
                {
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "label": point.get("label", self.label),
                    "conf": float(point.get("conf", 1.0)),
                }
                for point in points
            ]
        if normalized_method == "grid":
            return self.point_centroid_merge(points, dist_threshold=eps)
        if normalized_method == "dbscan":
            if len(points) > self.FAST_CLUSTER_THRESHOLD:
                return self.point_centroid_merge(points, dist_threshold=eps)
            return self.dbscan_filter(points, eps=eps, min_samples=min_samples)

        # Hybrid default: fast spatial pre-merge then DBSCAN for stable final centers.
        premerged = self.point_centroid_merge(
            points, dist_threshold=max(4.0, eps * 0.65)
        )
        if len(premerged) > self.FAST_CLUSTER_THRESHOLD:
            return self.point_centroid_merge(premerged, dist_threshold=eps)
        return self.dbscan_filter(premerged, eps=eps, min_samples=min_samples)

    def dbscan_filter(
        self,
        points: List[Dict[str, Any]],
        eps: float = 12.0,
        min_samples: int = 1,
    ) -> List[Dict[str, Any]]:
        """Filter nearby detections using DBSCAN clustering."""
        eps = max(0.1, float(eps))
        min_samples = max(1, int(min_samples))
        if len(points) < 2:
            return [
                {
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "label": point.get("label", self.label),
                    "conf": float(point.get("conf", 1.0)),
                }
                for point in points
            ]

        coords = np.array([[p["x"], p["y"]] for p in points])
        weights = np.array(
            [
                max(1e-6, float(p.get("conf", 1.0)) * float(p.get("edge_weight", 1.0)))
                for p in points
            ]
        )

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_

        unique_labels = set(labels)
        merged: List[Dict[str, Any]] = []

        for lbl in unique_labels:
            cluster_mask = labels == lbl
            cluster_points = [points[i] for i in range(len(points)) if cluster_mask[i]]
            cluster_weights = weights[cluster_mask]

            total_weight = cluster_weights.sum()
            if total_weight < 1e-6:
                total_weight = 1.0

            x_center = (
                np.sum(coords[cluster_mask][:, 0] * cluster_weights) / total_weight
            )
            y_center = (
                np.sum(coords[cluster_mask][:, 1] * cluster_weights) / total_weight
            )

            merged.append(
                {
                    "x": float(x_center),
                    "y": float(y_center),
                    "label": cluster_points[0].get("label", self.label),
                    "conf": float(max(p.get("conf", 1.0) for p in cluster_points)),
                }
            )

        return merged

    def point_centroid_merge(
        self,
        points: List[Dict[str, Any]],
        dist_threshold: float = 15.0,
    ) -> List[Dict[str, Any]]:
        if len(points) < 2:
            return [
                {
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "label": point.get("label", self.label),
                    "conf": float(point.get("conf", 1.0)),
                }
                for point in points
            ]

        cell_size = float(dist_threshold)
        grid: Dict[tuple[int, int], List[int]] = {}
        for index, point in enumerate(points):
            gx = int(point["x"] // cell_size)
            gy = int(point["y"] // cell_size)
            grid.setdefault((gx, gy), []).append(index)

        visited = [False] * len(points)
        merged: List[Dict[str, Any]] = []

        for start_idx in range(len(points)):
            if visited[start_idx]:
                continue

            queue = [start_idx]
            visited[start_idx] = True
            cluster_indices: List[int] = []
            cluster_label = points[start_idx].get("label", self.label)

            while queue:
                current_idx = queue.pop()
                cluster_indices.append(current_idx)
                current_point = points[current_idx]
                cx = current_point["x"]
                cy = current_point["y"]
                gx = int(cx // cell_size)
                gy = int(cy // cell_size)

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for next_idx in grid.get((gx + dx, gy + dy), []):
                            if visited[next_idx]:
                                continue

                            candidate = points[next_idx]
                            if candidate.get("label", self.label) != cluster_label:
                                continue

                            nx = candidate["x"]
                            ny = candidate["y"]
                            if math.hypot(cx - nx, cy - ny) <= dist_threshold:
                                visited[next_idx] = True
                                queue.append(next_idx)

            cluster = [points[index] for index in cluster_indices]
            weighted_sum = 0.0
            x_acc = 0.0
            y_acc = 0.0
            for point in cluster:
                confidence = float(point.get("conf", 1.0))
                edge_weight = float(point.get("edge_weight", 1.0))
                weight = max(1e-6, confidence * edge_weight)
                weighted_sum += weight
                x_acc += point["x"] * weight
                y_acc += point["y"] * weight

            merged.append(
                {
                    "x": x_acc / weighted_sum,
                    "y": y_acc / weighted_sum,
                    "label": cluster_label,
                    "conf": max(float(point.get("conf", 1.0)) for point in cluster),
                }
            )

        return merged

    def get_tile_coverage_map(
        self,
        tiff_path: str,
        tile_size: int = 640,
        overlap: float = 0.25,
        scan_window: Optional[List[float]] = None,
        polygon: Optional[List[List[float]]] = None,
        max_passes: int = 2,
    ) -> np.ndarray:
        poly_obj: Optional[Polygon] = None
        if polygon and len(polygon) >= 3:
            poly_obj = Polygon(polygon)

        with rasterio.open(tiff_path) as src:
            plan = self._plan_tile_passes(
                img_width=src.width,
                img_height=src.height,
                tile_size=tile_size,
                overlap=overlap,
                scan_window=scan_window,
                poly_obj=poly_obj,
                max_passes=max_passes,
            )
            return plan["gross_coverage"]

    def get_tile_coverage_stats(
        self,
        tiff_path: str,
        tile_size: int = 640,
        overlap: float = 0.25,
        scan_window: Optional[List[float]] = None,
        polygon: Optional[List[List[float]]] = None,
        max_passes: int = 2,
    ) -> Dict[str, Any]:
        poly_obj: Optional[Polygon] = None
        if polygon and len(polygon) >= 3:
            poly_obj = Polygon(polygon)

        with rasterio.open(tiff_path) as src:
            plan = self._plan_tile_passes(
                img_width=src.width,
                img_height=src.height,
                tile_size=tile_size,
                overlap=overlap,
                scan_window=scan_window,
                poly_obj=poly_obj,
                max_passes=max_passes,
            )

        coverage = plan["gross_coverage"]
        safe_coverage = plan["safe_coverage"]
        roi_mask = plan["roi_mask"]
        start_x = int(plan["start_x"])
        start_y = int(plan["start_y"])

        if coverage.size == 0:
            return {
                "has_gaps": True,
                "coverage_ratio": 0.0,
                "covered_pixels": 0,
                "uncovered_pixels": 0,
                "roi_pixels": 0,
                "min_coverage": 0,
                "max_coverage": 0,
                "safe_has_gaps": True,
                "safe_coverage_ratio": 0.0,
                "safe_covered_pixels": 0,
                "safe_uncovered_pixels": 0,
                "scan_window": scan_window,
            }

        roi_pixels = int(np.count_nonzero(roi_mask))
        if roi_pixels <= 0:
            roi_mask = np.ones(coverage.shape, dtype=bool)
            roi_pixels = int(np.count_nonzero(roi_mask))
            safe_coverage_mask = safe_coverage[roi_mask]
        else:
            safe_coverage_mask = safe_coverage[roi_mask]

        roi_coverage = coverage[roi_mask]
        covered_pixels = int(np.count_nonzero(roi_coverage > 0))
        uncovered_pixels = max(0, roi_pixels - covered_pixels)
        min_coverage = int(np.min(roi_coverage)) if roi_coverage.size else 0
        max_coverage = int(np.max(roi_coverage)) if roi_coverage.size else 0
        coverage_ratio = covered_pixels / roi_pixels if roi_pixels else 0.0
        safe_covered_pixels = int(np.count_nonzero(safe_coverage_mask > 0))
        safe_uncovered_pixels = max(0, roi_pixels - safe_covered_pixels)
        safe_coverage_ratio = safe_covered_pixels / roi_pixels if roi_pixels else 0.0

        return {
            "has_gaps": uncovered_pixels > 0,
            "coverage_ratio": coverage_ratio,
            "covered_pixels": covered_pixels,
            "uncovered_pixels": uncovered_pixels,
            "roi_pixels": roi_pixels,
            "min_coverage": min_coverage,
            "max_coverage": max_coverage,
            "safe_has_gaps": safe_uncovered_pixels > 0,
            "safe_coverage_ratio": safe_coverage_ratio,
            "safe_covered_pixels": safe_covered_pixels,
            "safe_uncovered_pixels": safe_uncovered_pixels,
            "scan_window": [start_x, start_y, coverage.shape[1], coverage.shape[0]],
        }

    @staticmethod
    def validate_model_path(model_path: str) -> Dict[str, Any]:
        model_path = os.path.abspath(model_path)
        response: Dict[str, Any] = {
            "path": model_path,
            "exists": os.path.exists(model_path),
            "valid": False,
            "error": None,
        }

        if not response["exists"]:
            response["error"] = "Model file does not exist"
            return response

        try:
            YOLO(model_path)
            response["valid"] = True
        except Exception as exc:
            response["error"] = str(exc)

        return response


class MultiModelDetector:
    def __init__(self, model_entries: Optional[List[Dict[str, Any]]] = None):
        self.model_entries: List[Dict[str, Any]] = []
        self.models: Dict[str, TreeDetector] = {}
        self.model_errors: Dict[str, str] = {}
        if model_entries:
            self.reload(model_entries)

    def reload(self, model_entries: List[Dict[str, Any]]) -> None:
        self.model_entries = model_entries or []
        self.models = {}
        self.model_errors = {}

        for entry in self.model_entries:
            model_id = str(entry.get("id", "")).strip()
            if not model_id:
                continue

            if not bool(entry.get("enabled", True)):
                continue

            model_path = str(entry.get("path", "")).strip()
            if not model_path:
                self.model_errors[model_id] = "Missing model path"
                continue

            try:
                detector = TreeDetector(
                    model_path=os.path.abspath(model_path),
                    model_id=model_id,
                    label=str(entry.get("label", model_id)),
                    name=str(entry.get("name", model_id)),
                    default_conf=float(entry.get("conf", 0.1)),
                    default_imgsz=int(entry.get("imgsz", 640)),
                )
                self.models[model_id] = detector
            except Exception as exc:
                self.model_errors[model_id] = str(exc)

    def get_loaded_model_ids(self) -> List[str]:
        return list(self.models.keys())

    def get_model_errors(self) -> Dict[str, str]:
        return dict(self.model_errors)

    def _resolve_models(
        self, selected_model_ids: Optional[Iterable[str]]
    ) -> List[TreeDetector]:
        if selected_model_ids:
            ordered_unique_ids: List[str] = []
            for model_id in selected_model_ids:
                if model_id not in ordered_unique_ids:
                    ordered_unique_ids.append(model_id)
            selected = [
                self.models[mid] for mid in ordered_unique_ids if mid in self.models
            ]
        else:
            selected = list(self.models.values())

        if not selected:
            raise ValueError("No valid models loaded for inference")
        return selected

    @staticmethod
    def _merge_same_label_points(
        points: List[Dict[str, Any]], dist_threshold: float
    ) -> List[Dict[str, Any]]:
        if len(points) < 2:
            return [
                {
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "label": str(point.get("label", "palm_tree")),
                    "conf": float(point.get("conf", 1.0)),
                }
                for point in points
            ]

        cell_size = float(dist_threshold)
        grid: Dict[tuple[int, int], List[int]] = {}
        for index, point in enumerate(points):
            gx = int(point["x"] // cell_size)
            gy = int(point["y"] // cell_size)
            grid.setdefault((gx, gy), []).append(index)

        visited = [False] * len(points)
        merged: List[Dict[str, Any]] = []

        for start_idx in range(len(points)):
            if visited[start_idx]:
                continue

            queue = [start_idx]
            visited[start_idx] = True
            cluster_indices: List[int] = []

            while queue:
                current_idx = queue.pop()
                cluster_indices.append(current_idx)
                current_point = points[current_idx]
                cx = current_point["x"]
                cy = current_point["y"]
                gx = int(cx // cell_size)
                gy = int(cy // cell_size)

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for next_idx in grid.get((gx + dx, gy + dy), []):
                            if visited[next_idx]:
                                continue
                            candidate = points[next_idx]
                            nx = candidate["x"]
                            ny = candidate["y"]
                            if math.hypot(cx - nx, cy - ny) <= dist_threshold:
                                visited[next_idx] = True
                                queue.append(next_idx)

            cluster = [points[index] for index in cluster_indices]
            weighted_sum = 0.0
            x_acc = 0.0
            y_acc = 0.0
            for point in cluster:
                confidence = float(point.get("conf", 1.0))
                edge_weight = float(point.get("edge_weight", 1.0))
                weight = max(1e-6, confidence * edge_weight)
                weighted_sum += weight
                x_acc += float(point["x"]) * weight
                y_acc += float(point["y"]) * weight

            merged.append(
                {
                    "x": x_acc / weighted_sum,
                    "y": y_acc / weighted_sum,
                    "label": cluster[0].get("label", "palm_tree"),
                    "conf": max(float(point.get("conf", 1.0)) for point in cluster),
                }
            )

        return merged

    @staticmethod
    def _resolve_cross_label_conflicts(
        points: List[Dict[str, Any]], dist_threshold: float
    ) -> List[Dict[str, Any]]:
        sorted_points = sorted(
            points, key=lambda point: float(point.get("conf", 0.0)), reverse=True
        )
        kept: List[Dict[str, Any]] = []

        for point in sorted_points:
            px = float(point["x"])
            py = float(point["y"])
            plabel = str(point.get("label", "palm_tree"))

            has_conflict = False
            for existing in kept:
                if str(existing.get("label", "palm_tree")) == plabel:
                    continue
                ex = float(existing["x"])
                ey = float(existing["y"])
                if math.hypot(px - ex, py - ey) <= dist_threshold:
                    has_conflict = True
                    break

            if not has_conflict:
                kept.append(
                    {
                        "x": px,
                        "y": py,
                        "label": plabel,
                        "conf": float(point.get("conf", 1.0)),
                    }
                )

        return kept

    def merge_points(
        self,
        points: List[Dict[str, Any]],
        same_label_dist: float = 12.0,
        cross_label_dist: float = 10.0,
    ) -> List[Dict[str, Any]]:
        if not points:
            return []

        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for point in points:
            label = str(point.get("label", "palm_tree"))
            by_label.setdefault(label, []).append(point)

        merged_same_label: List[Dict[str, Any]] = []
        for label_points in by_label.values():
            merged_same_label.extend(
                self._merge_same_label_points(
                    label_points, dist_threshold=same_label_dist
                )
            )

        deduped = self._resolve_cross_label_conflicts(
            merged_same_label, dist_threshold=cross_label_dist
        )
        deduped.sort(key=lambda point: (point["y"], point["x"]))
        return deduped

    def predict_tiff_slice(
        self,
        tiff_path: str,
        x: int,
        y: int,
        w: int,
        h: int,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        selected_model_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        models = self._resolve_models(selected_model_ids)

        all_points: List[Dict[str, Any]] = []
        for model in models:
            model_conf = float(conf) if conf is not None else model.default_conf
            model_imgsz = int(imgsz) if imgsz is not None else model.default_imgsz
            all_points.extend(
                model.predict_tiff_slice(
                    tiff_path=tiff_path,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    conf=model_conf,
                    imgsz=model_imgsz,
                )
            )

        return self.merge_points(all_points)

    def predict_tiff_full(
        self,
        tiff_path: str,
        conf: Optional[float] = None,
        tile_size: int = 640,
        overlap: float = 0.25,
        scan_window: Optional[List[float]] = None,
        polygon: Optional[List[List[float]]] = None,
        imgsz: Optional[int] = None,
        selected_model_ids: Optional[Iterable[str]] = None,
        dbscan_eps: float = 12.0,
        cluster_method: str = "hybrid",
        max_passes: int = 2,
        min_new_points: int = 2,
        batch_size: int = 8,
        zoom_scales: Optional[Iterable[float]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        models = self._resolve_models(selected_model_ids)
        total_models = len(models)
        collected_points: List[Dict[str, Any]] = []

        for model_index, model in enumerate(models):
            model_conf = float(conf) if conf is not None else model.default_conf
            model_imgsz = int(imgsz) if imgsz is not None else model.default_imgsz

            for update in model.predict_tiff_full(
                tiff_path=tiff_path,
                conf=model_conf,
                tile_size=tile_size,
                overlap=overlap,
                scan_window=scan_window,
                polygon=polygon,
                imgsz=model_imgsz,
                dbscan_eps=dbscan_eps,
                cluster_method=cluster_method,
                max_passes=max_passes,
                min_new_points=min_new_points,
                batch_size=batch_size,
                zoom_scales=zoom_scales,
            ):
                update_type = update.get("type")

                if update_type == "window":
                    local_progress = float(update.get("progress", 0.0))
                    update["progress"] = (model_index + local_progress) / total_models
                    update["model_id"] = model.model_id
                    yield update
                    continue

                if update_type == "points":
                    points = update.get("points", [])
                    for point in points:
                        point.setdefault("model_id", model.model_id)
                    yield {
                        "type": "points",
                        "points": points,
                        "model_id": model.model_id,
                    }
                    continue

                if update_type == "final":
                    collected_points.extend(update.get("points", []))

        final_points = self.merge_points(collected_points)
        yield {"type": "final", "points": final_points}
