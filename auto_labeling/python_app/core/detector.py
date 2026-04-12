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
            source=image_source,
            conf=conf,
            imgsz=imgsz,
            augment=True,
            verbose=False
        )
        points = []
        for r in results:
            boxes = r.boxes.xywh.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                points.append({
                    "x": float(box[0]),
                    "y": float(box[1]),
                    "label": "palm_tree",
                    "conf": float(confs[i]) if i < len(confs) else 1.0
                })
        return points

    def predict_tiff_slice(self, tiff_path, x, y, w, h, conf=0.01):
        with rasterio.open(tiff_path) as src:
            window = rasterio.windows.Window(x, y, w, h)
            num_bands = min(src.count, 3)
            data = src.read(list(range(1, num_bands + 1)), window=window)
            if num_bands == 1:
                data = np.repeat(data, 3, axis=0)
            elif num_bands == 2:
                data = np.concatenate([data, np.zeros((1, h, w), dtype=data.dtype)], axis=0)
            data = np.transpose(data, (1, 2, 0))
            results = self.model.predict(source=data, conf=conf, imgsz=1280, augment=True, verbose=False)
            points = []
            for r in results:
                boxes = r.boxes.xywh.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for i, box in enumerate(boxes):
                    points.append({
                        "x": float(box[0] + x),
                        "y": float(box[1] + y),
                        "label": "palm_tree",
                        "conf": float(confs[i]) if i < len(confs) else 1.0
                    })
            return points

    def predict_tiff_full(self, tiff_path, conf=0.1, tile_size=640, overlap=0.25, scan_window=None, polygon=None):
        """
        High-performance SAHI inference with streaming.
        polygon: List of [x, y] coordinates forming a polygon.
        """
        from shapely.geometry import Polygon, Point
        
        effective_conf = max(float(conf), 0.1)
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

            y_starts = sorted(list(set(list(range(start_y, max(start_y + 1, start_y + scan_h - tile_size), stride)) + [max(start_y, start_y + scan_h - tile_size)])))
            x_starts = sorted(list(set(list(range(start_x, max(start_x + 1, start_x + scan_w - tile_size), stride)) + [max(start_x, start_x + scan_w - tile_size)])))

            total_tiles = len(y_starts) * len(x_starts)
            current_tile = 0

            for y_pos in y_starts:
                for x_pos in x_starts:
                    current_tile += 1
                    w = min(tile_size, start_x + scan_w - x_pos)
                    h = min(tile_size, start_y + scan_h - y_pos)
                    if w < 10 or h < 10: continue

                    if poly_obj:
                        tile_poly = Polygon([(x_pos, y_pos), (x_pos+w, y_pos), (x_pos+w, y_pos+h), (x_pos, y_pos+h)])
                        if not poly_obj.intersects(tile_poly): continue

                    window = rasterio.windows.Window(x_pos, y_pos, w, h)
                    num_bands = min(src.count, 3)
                    data = src.read(list(range(1, num_bands + 1)), window=window)
                    
                    if num_bands == 1:
                        data = np.repeat(data, 3, axis=0)
                    elif num_bands == 2:
                        data = np.concatenate([data, np.zeros((1, h, w), dtype=data.dtype)], axis=0)
                    data = np.transpose(data, (1, 2, 0))

                    if data.dtype != np.uint8:
                        sample = data[::4, ::4]
                        p2, p98 = np.percentile(sample, (2, 98))
                        if p98 > p2:
                            data = np.clip(data, p2, p98)
                            data = ((data - p2) / (p98 - p2) * 255).astype(np.uint8)
                        else: data = data.astype(np.uint8)

                    input_data = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    input_data[0:h, 0:w, :] = data

                    yield {"type": "window", "window": [x_pos, y_pos, w, h], "progress": current_tile / total_tiles}

                    if input_data.mean() > 5:
                        # Optimization: Disable augment for SAHI scan to gain speed
                        results = self.model.predict(
                            source=input_data, 
                            conf=effective_conf, 
                            imgsz=tile_size, 
                            verbose=False,
                            augment=False 
                        )
                        tile_points = []
                        for r in results:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            for i, box in enumerate(boxes):
                                tx, ty = (box[0] + box[2])/2 + x_pos, (box[1] + box[3])/2 + y_pos
                                if poly_obj and not poly_obj.contains(Point(tx, ty)): continue
                                if (x_pos + tile_size < start_x + scan_w and tx > x_pos + stride + 5) or \
                                   (y_pos + tile_size < start_y + scan_h and ty > y_pos + stride + 5):
                                    continue
                                p = {"x": float(tx), "y": float(ty), "label": "palm_tree", "conf": float(confs[i])}
                                tile_points.append(p)
                                all_points_raw.append(p)
                        if tile_points:
                            yield {"type": "points", "points": tile_points}

            if all_points_raw:
                final_points = self.point_centroid_merge(all_points_raw, dist_threshold=12)
                yield {"type": "final", "points": final_points}
            else:
                yield {"type": "final", "points": []}

    def point_centroid_merge(self, points, dist_threshold=15):
        if len(points) < 2: return points
        pts = sorted(points, key=lambda p: p['x'])
        merged = []
        used = [False] * len(pts)
        for i in range(len(pts)):
            if used[i]: continue
            cluster = [pts[i]]
            used[i] = True
            for j in range(i + 1, len(pts)):
                if used[j]: continue
                if pts[j]['x'] - pts[i]['x'] > dist_threshold: break
                if abs(pts[j]['y'] - pts[i]['y']) > dist_threshold: continue
                if math.hypot(pts[i]['x'] - pts[j]['x'], pts[i]['y'] - pts[j]['y']) < dist_threshold:
                    cluster.append(pts[j])
                    used[j] = True
            merged.append({
                "x": sum(p['x'] for p in cluster) / len(cluster),
                "y": sum(p['y'] for p in cluster) / len(cluster),
                "label": cluster[0]['label'],
                "conf": max(p['conf'] for p in cluster)
            })
        return merged
