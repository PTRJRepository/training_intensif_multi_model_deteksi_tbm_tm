from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon


PYTHON_APP_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_APP_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_APP_DIR))

from core.detector import TreeDetector


def test_polygon_boundary_points_are_included():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    assert TreeDetector._point_in_polygon(poly, 10, 5)


def test_low_mean_tiles_are_still_processed():
    tile = np.full((640, 640, 3), 2, dtype=np.uint8)

    assert TreeDetector._should_process_tile(tile)


def _axis_coverage_is_complete(
    starts: list[int], start: int, end: int, tile_size: int
) -> bool:
    if not starts:
        return False

    intervals = sorted((s, min(end, s + tile_size)) for s in starts)
    cursor = start
    for left, right in intervals:
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return cursor >= end


def test_axis_starts_cover_roi_without_offset():
    start, end = 0, 2000
    tile_size = 640
    stride = 480
    starts = TreeDetector._compute_axis_starts(
        start=start, end=end, tile_size=tile_size, stride=stride, offset=0
    )

    assert starts[0] == start
    assert starts[-1] == end - tile_size
    assert _axis_coverage_is_complete(starts, start, end, tile_size)


def test_axis_starts_cover_roi_with_offset():
    start, end = 153, 2137
    tile_size = 640
    stride = 480
    starts = TreeDetector._compute_axis_starts(
        start=start, end=end, tile_size=tile_size, stride=stride, offset=240
    )

    assert starts[0] == start
    assert starts[-1] == end - tile_size
    assert _axis_coverage_is_complete(starts, start, end, tile_size)


def test_tile_coverage_stats_detect_full_roi_coverage(monkeypatch):
    detector = TreeDetector.__new__(TreeDetector)

    monkeypatch.setattr(
        TreeDetector,
        "get_tile_coverage_map",
        lambda self, *args, **kwargs: np.ones((6, 8), dtype=np.float32),
    )

    stats = TreeDetector.get_tile_coverage_stats(
        detector,
        "dummy.tif",
        scan_window=[10, 20, 8, 6],
    )

    assert stats["has_gaps"] is False
    assert stats["covered_pixels"] == 48
    assert stats["uncovered_pixels"] == 0
    assert stats["coverage_ratio"] == 1.0


def test_tile_coverage_stats_detect_polygon_gaps(monkeypatch):
    detector = TreeDetector.__new__(TreeDetector)
    coverage = np.ones((10, 10), dtype=np.float32)
    coverage[8:, 8:] = 0

    monkeypatch.setattr(
        TreeDetector,
        "get_tile_coverage_map",
        lambda self, *args, **kwargs: coverage,
    )

    stats = TreeDetector.get_tile_coverage_stats(
        detector,
        "dummy.tif",
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
    )

    assert stats["has_gaps"] is True
    assert stats["uncovered_pixels"] > 0
    assert stats["coverage_ratio"] < 1.0


def test_plan_tile_passes_expands_polygon_scan_window():
    detector = TreeDetector.__new__(TreeDetector)
    poly = Polygon([(400, 400), (900, 400), (900, 900), (400, 900)])

    plan = TreeDetector._plan_tile_passes(
        detector,
        img_width=2000,
        img_height=2000,
        tile_size=640,
        overlap=0.25,
        scan_window=None,
        poly_obj=poly,
        max_passes=1,
    )

    assert plan["start_x"] < 400
    assert plan["start_y"] < 400
    assert plan["safe_gap_pixels_by_pass"][-1] == 0


def test_generate_gap_patch_jobs_adds_tiles_for_safe_gap():
    detector = TreeDetector.__new__(TreeDetector)
    gap_mask = np.zeros((400, 400), dtype=bool)
    gap_mask[180:220, 190:230] = True
    poly = Polygon([(100, 100), (700, 100), (700, 700), (100, 700)])

    jobs = TreeDetector._generate_gap_patch_jobs(
        detector,
        gap_mask=gap_mask,
        start_x=100,
        start_y=100,
        img_width=2000,
        img_height=2000,
        tile_size=640,
        poly_obj=poly,
        existing_jobs=[],
    )

    assert jobs
    x_pos, y_pos, w, h = jobs[0]
    assert x_pos <= 310 <= x_pos + w
    assert y_pos <= 300 <= y_pos + h
