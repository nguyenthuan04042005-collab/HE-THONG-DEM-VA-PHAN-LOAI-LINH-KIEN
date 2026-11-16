"""Regression tests for Canny geometry, object counting, and evaluation."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source_code"))
sys.path.insert(0, str(ROOT))
from canny import detect_edges
from classifier import classify
from config import Config
from evaluate import box_iou, match_objects, evaluate
from filters import calculate_gradient, apply_masks
from pipeline import analyze_image
from processing import non_maxima_suppression, hysteresis_thresholding
from segmentation import fill_enclosed_regions, connected_components, segment_objects
from scripts.generate_synthetic import make_scene


class CannyTests(unittest.TestCase):
    def test_gradient_axes_are_consistent(self):
        gx, gy, scale = calculate_gradient(7, 1.0)
        rows = np.repeat(np.arange(21)[:, None], 21, axis=1)
        fx, fy = apply_masks(rows, gx, gy, scale)
        self.assertEqual(fx[10, 10], 0)
        self.assertNotEqual(fy[10, 10], 0)
        fx, fy = apply_masks(rows.T, gx, gy, scale)
        self.assertNotEqual(fx[10, 10], 0)
        self.assertEqual(fy[10, 10], 0)

    def test_nms_compares_correct_diagonals(self):
        for direction, neighbors in [(1, ((1, 1), (3, 3))), (3, ((1, 3), (3, 1)))]:
            magnitude = np.zeros((5, 5))
            magnitude[2, 2] = 5
            for point in neighbors:
                magnitude[point] = 9
            directions = np.full((5, 5), direction, dtype=np.uint8)
            self.assertEqual(non_maxima_suppression(magnitude, directions)[0][2, 2], 0)

    def test_constant_images_have_no_edges(self):
        for intensity in (0, 128, 255):
            edges, _ = detect_edges(np.full((40, 50), intensity, dtype=np.uint8))
            self.assertFalse(edges.any())

    def test_hysteresis_keeps_only_connected_weak_pixels(self):
        magnitude = np.zeros((8, 12))
        magnitude[4, 2] = 100
        magnitude[4, 3:10] = 50
        magnitude[1, 8] = 50
        edges = hysteresis_thresholding(magnitude, 200, 100)
        self.assertEqual(np.count_nonzero(edges), 8)
        self.assertEqual(edges[1, 8], 0)

    def test_invalid_zero_threshold_rejected(self):
        with self.assertRaises(ValueError):
            hysteresis_thresholding(np.zeros((10, 10)), 100, 0)
        with self.assertRaises(ValueError):
            Config(low_threshold=0)


class SegmentationTests(unittest.TestCase):
    def test_nested_boundaries_form_one_object(self):
        edges = np.zeros((40, 40), dtype=bool)
        for lo, hi in ((5, 34), (14, 25)):
            edges[lo, lo:hi + 1] = edges[hi, lo:hi + 1] = True
            edges[lo:hi + 1, lo] = edges[lo:hi + 1, hi] = True
        filled = fill_enclosed_regions(edges)
        self.assertTrue(filled[20, 20])
        self.assertFalse(filled[0, 0])
        _, regions = connected_components(filled)
        self.assertEqual(len(regions), 1)

    def test_open_boundary_is_not_filled(self):
        edges = np.zeros((40, 40), dtype=bool)
        edges[5:35, 5] = edges[5:35, 34] = True
        edges[5, 5:35] = True
        self.assertFalse(fill_enclosed_regions(edges)[20, 20])

    def test_diagonal_connectivity_is_explicit(self):
        mask = np.eye(3, dtype=bool)
        self.assertEqual(len(connected_components(mask, 4)[1]), 3)
        self.assertEqual(len(connected_components(mask, 8)[1]), 1)

    def test_small_noise_rejected(self):
        edges = np.zeros((50, 50), dtype=np.uint8)
        edges[15, 15] = 255
        components, _, _, _ = segment_objects(edges, Config(min_area=20, closing_iterations=0))
        self.assertEqual(components, [])


class SystemTests(unittest.TestCase):
    def test_pipeline_counts_varied_rotations_and_sizes(self):
        with tempfile.TemporaryDirectory() as folder:
            for seed in (2, 17, 41):
                with self.subTest(seed=seed):
                    image, _ = make_scene(seed)
                    path = Path(folder) / f"scene_{seed}.png"
                    image.save(path)
                    report, _ = analyze_image(path)
                    self.assertEqual(report["counts"], {"bolt": 2, "washer": 2, "nut": 2, "unknown": 0})

    def test_blank_image_counts_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blank.png"
            Image.new("RGB", (100, 100), "white").save(path)
            report, _ = analyze_image(path)
            self.assertEqual(report["total"], 0)

    def test_ambiguous_shape_is_unknown(self):
        shape = {"elongation": 1.0, "hole_count": 0, "radial_cv": 0.01, "hex_score": 0.01}
        self.assertEqual(classify(shape, Config())[0], "unknown")

    def test_color_polarity_does_not_change_counts(self):
        image, _ = make_scene(8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inverted.png"
            Image.fromarray(255 - np.asarray(image)).save(path)
            report, _ = analyze_image(path)
            self.assertEqual(report["counts"], {"bolt": 2, "washer": 2, "nut": 2, "unknown": 0})


class EvaluationTests(unittest.TestCase):
    def test_matching_is_one_to_one(self):
        truth = [{"label": "nut", "bbox": [0, 0, 10, 10]}]
        predicted = [{"label": "nut", "bbox": [0, 0, 10, 10]}] * 2
        matches, missed, spurious = match_objects(truth, predicted)
        self.assertEqual((len(matches), len(missed), len(spurious)), (1, 0, 1))
        self.assertEqual(box_iou([0, 0, 5, 5], [10, 10, 15, 15]), 0)

    def test_classification_error_is_not_hidden_by_detection(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            annotation = Path(folder) / "labels.json"
            annotation.write_text(json.dumps({"images": [{"image": "unused.png", "objects": [
                {"label": "nut", "bbox": [5, 5, 15, 15]}]}]}), encoding="utf-8")
            fake_report = {"image_size": [20, 20], "counts": {"bolt": 0, "washer": 1, "nut": 0, "unknown": 0},
                           "objects": [{"label": "washer", "bbox": [5, 5, 15, 15]}]}
            with patch("evaluate.analyze_image", return_value=(fake_report, {})), patch("evaluate.save_result"):
                metrics = evaluate(annotation, Path(folder) / "evaluation", Config())
            self.assertEqual(metrics["detection_recall"], 1)
            self.assertEqual(metrics["classification_accuracy_on_matched"], 0)
            self.assertEqual(metrics["confusion_matrix"]["nut"]["washer"], 1)
            self.assertEqual(metrics["class_metrics"]["nut"]["fn"], 1)
            self.assertEqual(metrics["class_metrics"]["washer"]["fp"], 1)


if __name__ == "__main__":
    unittest.main()
