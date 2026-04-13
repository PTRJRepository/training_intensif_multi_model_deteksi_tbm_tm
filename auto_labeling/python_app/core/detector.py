from ultralytics import YOLO
import numpy as np
import rasterio
from PIL import Image
import os
import math


class TreeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict_image(self, image_source, conf=0.01, imgsz=640):
        results = self.model.predict(
            source=image_source, conf=conf, imgsz=imgsz, augment=True, verbose=False
        )
        points = []
        for r in results:
            boxes = r.boxes.xywh.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                points.append(
                    {
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "label": "palm_tree",
                        "conf": float(confs[i]) if i < len(confs) else 1.0,
                    }
                )
        return points

    def predict_tiff_slice(self, tiff_path, x, y, w, h, conf=0.01, imgsz=640):
        with rasterio.open(tiff_path) as src:
            window = rasterio.windows.Window(x, y, w, h)
            num_bands = min(src.count, 3)
            data = src.read(list(range(1, num_bands + 1)), window=window)
            if num_bands == 1:
                data = np.repeat(data, 3, axis=0)
            elif num_bands == 2:
                data = np.concatenate(
                    [data, np.zeros((1, h, w), dtype=data.dtype)], axis=0
                )
            data = np.transpose(data, (1, 2, 0))
            results = self.model.predict(
                source=data, conf=conf, imgsz=imgsz, augment=True, verbose=False
            )
            points = []
            for r in results:
                boxes = r.boxes.xywh.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for i, box in enumerate(boxes):
                    points.append(
                        {
                            "x": float(box[0] + x),
                            "y": float(box[1] + y),
                            "label": "palm_tree",
                            "conf": float(confs[i]) if i < len(confs) else 1.0,
                        }
                    )
            return points

    def predict_tiff_full(
        self,
        tiff_path,
        conf=0.1,
        tile_size=640,
        overlap=0.25,
        scan_window=None,
        polygon=None,
        imgsz=640,
    ):
        """
        High-performance SAHI inference with streaming.
        polygon: List of [x, y] coordinates forming a polygon.
        tile_size: Size of scanning window (default 640)
        imgsz: YOLO inference size (default 640, should match training size)
        """
        from shapely.geometry import Polygon, Point

        effective_conf = float(conf)
        poly_obj = None
        if polygon and len(polygon) >= 3:
            poly_obj = Polygon(polygon)
            minx, miny, maxx, maxy = poly_obj.bounds
            scan_window = [minx, miny, maxx - minx, maxy - miny]

        with rasterio.open(tiff_path) as src:
            img_width, img_height = src.width, src.height
            all_points_raw = []

            if scan_window:
                sx, sy, sw, sh = scan_window
                start_x = max(0, int(sx))
                start_y = max(0, int(sy))
                end_x = min(img_width, int(sx + sw))
                end_y = min(img_height, int(sy + sh))
                scan_w = end_x - start_x
                scan_h = end_y - start_y
            else:
                start_x, start_y = 0, 0
                scan_w, scan_h = img_width, img_height

            overlap = min(0.9, max(0, overlap))
            stride = max(10, tile_size - int(tile_size * overlap))

            y_starts = sorted(
                list(
                    set(
                        list(
                            range(
                                start_y,
                                max(start_y + 1, start_y + scan_h - tile_size),
                                stride,
                            )
                        )
                        + [max(start_y, start_y + scan_h - tile_size)]
                    )
                )
            )
            x_starts = sorted(
                list(
                    set(
                        list(
                            range(
                                start_x,
                                max(start_x + 1, start_x + scan_w - tile_size),
                                stride,
                            )
                        )
                        + [max(start_x, start_x + scan_w - tile_size)]
                    )
                )
            )

            total_tiles = len(y_starts) * len(x_starts)
            current_tile = 0

            for y_pos in y_starts:
                for x_pos in x_starts:
                    current_tile += 1
                    w = min(tile_size, start_x + scan_w - x_pos)
                    h = min(tile_size, start_y + scan_h - y_pos)
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

                    yield {
                        "type": "window",
                        "window": [x_pos, y_pos, w, h],
                        "progress": current_tile / total_tiles,
                    }

                    if input_data.mean() > 5:
                        # Optimization: Disable augment for SAHI scan to gain speed
                        results = self.model.predict(
                            source=input_data,
                            conf=effective_conf,
                            imgsz=imgsz,
                            verbose=False,
                            augment=False,
                        )
                        tile_points = []
                        edge_band = max(8.0, tile_size * overlap * 0.5)
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
                                tx = cx_local + x_pos
                                ty = cy_local + y_pos

                                edge_margin = min(
                                    cx_local,
                                    cy_local,
                                    max(0.0, w - cx_local),
                                    max(0.0, h - cy_local),
                                )
                                edge_weight = max(
                                    0.2, min(1.0, edge_margin / edge_band)
                                )

                                if poly_obj and not poly_obj.contains(Point(tx, ty)):
                                    continue
                                p = {
                                    "x": float(tx),
                                    "y": float(ty),
                                    "label": "palm_tree",
                                    "conf": float(confs[i]),
                                    "edge_weight": float(edge_weight),
                                }
                                tile_points.append(p)
                                all_points_raw.append(p)
                        if tile_points:
                            yield {"type": "points", "points": tile_points}

            if all_points_raw:
                final_points = self.point_centroid_merge(
                    all_points_raw, dist_threshold=12
                )
                yield {"type": "final", "points": final_points}
            else:
                yield {"type": "final", "points": []}

    def point_centroid_merge(self, points, dist_threshold=15):
        if len(points) < 2:
            return [
                {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "label": p.get("label", "palm_tree"),
                    "conf": float(p.get("conf", 1.0)),
                }
                for p in points
            ]

        cell_size = float(dist_threshold)
        grid = {}
        for idx, p in enumerate(points):
            gx = int(p["x"] // cell_size)
            gy = int(p["y"] // cell_size)
            grid.setdefault((gx, gy), []).append(idx)

        visited = [False] * len(points)
        merged = []

        for i in range(len(points)):
            if visited[i]:
                continue

            queue = [i]
            visited[i] = True
            cluster_indices = []

            while queue:
                cur = queue.pop()
                cluster_indices.append(cur)
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

            cluster = [points[idx] for idx in cluster_indices]
            weighted_sum = 0.0
            x_acc = 0.0
            y_acc = 0.0
            for p in cluster:
                conf = float(p.get("conf", 1.0))
                edge_weight = float(p.get("edge_weight", 1.0))
                w = max(1e-6, conf * edge_weight)
                weighted_sum += w
                x_acc += p["x"] * w
                y_acc += p["y"] * w

            merged.append(
                {
                    "x": x_acc / weighted_sum,
                    "y": y_acc / weighted_sum,
                    "label": cluster[0].get("label", "palm_tree"),
                    "conf": max(float(p.get("conf", 1.0)) for p in cluster),
                }
            )

        return merged
