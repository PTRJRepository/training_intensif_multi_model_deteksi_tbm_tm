
import unittest
from unittest.mock import MagicMock
import numpy as np
from auto_labeling.python_app.core.detector import TreeDetector

class TestDetectorClipping(unittest.TestCase):
    def setUp(self):
        # Create a detector with a dummy model path
        # We'll mock the YOLO model anyway
        self.detector = TreeDetector.__new__(TreeDetector)
        self.detector.model_id = "test_model"
        self.detector.label = "palm_tree"
        self.detector.MIN_BOX_SIZE_PX = 2.0
        self.detector.MAX_BOX_SIZE_PX = 160.0
        self.detector.MAX_BOX_ASPECT_RATIO = 2.75
        self.detector._predict_kwargs = {}
        self.detector.model = MagicMock()

    def test_predict_batched_jobs_clipping(self):
        # Mocking the predict result
        # Imagine a tile of 640x640 at zoom 2.0 -> scaled tile is 1280x1280
        # A detection at (800, 800) in the scaled tile should NOT be clipped to 640.
        
        mock_box = MagicMock()
        # xyxy format: [left, top, right, bottom]
        # A box centered at 800, 800 with size 40x40 -> [780, 780, 820, 820]
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[780, 780, 820, 820]])
        mock_box.conf = MagicMock()
        mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        
        mock_result = MagicMock()
        mock_result.boxes = mock_box
        
        self.detector.model.predict.return_value = [mock_result]
        
        # Inputs
        batched_inputs = [np.zeros((1280, 1280, 3), dtype=np.uint8)]
        batched_jobs = [(1000, 1000, 640, 640)] # x_pos, y_pos, w, h
        scale_factor = 2.0
        effective_conf = 0.1
        imgsz = 1280
        edge_band = 80.0
        poly_obj = None
        
        points = self.detector._predict_batched_jobs(
            batched_inputs,
            batched_jobs,
            scale_factor,
            effective_conf,
            imgsz,
            edge_band,
            poly_obj
        )
        
        self.assertEqual(len(points), 1)
        point = points[0]
        
        # Expected point location:
        # local_center = (800, 800)
        # tx = (800 / 2.0) + 1000 = 400 + 1000 = 1400
        # ty = (800 / 2.0) + 1000 = 400 + 1000 = 1400
        self.assertEqual(point['x'], 1400.0)
        self.assertEqual(point['y'], 1400.0)
        
        # Check that it wasn't clipped to 640 before scaling
        # If it was clipped to 640:
        # left=640, right=640 -> cx_local=640
        # tx = (640 / 2.0) + 1000 = 320 + 1000 = 1320
        self.assertNotEqual(point['x'], 1320.0)

    def test_predict_batched_jobs_no_poly_filter(self):
        # Verify that if poly_obj is provided, it correctly filters points
        # Case 1: Point inside poly
        # Case 2: Point outside poly
        from shapely.geometry import Polygon
        poly = Polygon([(1000, 1000), (2000, 1000), (2000, 2000), (1000, 2000)])
        
        # Mock result at 1400, 1400 (inside)
        mock_box = MagicMock()
        mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[780, 780, 820, 820]])
        mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_result = MagicMock()
        mock_result.boxes = mock_box
        self.detector.model.predict.return_value = [mock_result]
        
        points = self.detector._predict_batched_jobs(
            [None], [(1000, 1000, 640, 640)], 2.0, 0.1, 1280, 80.0, poly
        )
        self.assertEqual(len(points), 1)
        
        # Mock result at 2400, 2400 (outside)
        # cx_local = 2800 -> tx = 1400 + 1000 = 2400
        mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[2780, 2780, 2820, 2820]])
        points = self.detector._predict_batched_jobs(
            [None], [(1000, 1000, 640, 640)], 2.0, 0.1, 1280, 80.0, poly
        )
        self.assertEqual(len(points), 0)

if __name__ == "__main__":
    unittest.main()
