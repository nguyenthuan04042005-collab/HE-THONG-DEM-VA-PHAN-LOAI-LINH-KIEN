"""Verify that swapped classes and unknown regions cannot inflate accuracy."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_dataset import compare


class CountBenchmarkTests(unittest.TestCase):
    def test_equal_total_is_not_equal_class_counts_and_unknown_is_an_error(self):
        entries = [
            {"name": "swapped.jpg", "counts": {"bolt": 0, "washer": 1, "nut": 1, "unknown": 0}, "total": 2, "seconds": 1},
            {"name": "extra_unknown.jpg", "counts": {"bolt": 0, "washer": 1, "nut": 0, "unknown": 1}, "total": 2, "seconds": 1},
            {"name": "correct.jpg", "counts": {"bolt": 0, "washer": 0, "nut": 0, "unknown": 0}, "total": 0, "seconds": 1},
        ]
        annotations = {"images": [
            {"image": "swapped.jpg", "counts": {"bolt": 1, "washer": 0, "nut": 1}},
            {"image": "extra_unknown.jpg", "counts": {"bolt": 0, "washer": 1, "nut": 0}},
            {"image": "correct.jpg", "counts": {"bolt": 0, "washer": 0, "nut": 0}},
        ]}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "predictions.json").write_text(json.dumps({"parameters": {}, "images": entries}), encoding="utf-8")
            label_path = folder / "labels.json"
            label_path.write_text(json.dumps(annotations), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = compare(folder, label_path)
            self.assertEqual(metrics["exact_class_counts_images"], 1)
            self.assertAlmostEqual(metrics["exact_class_counts_percent"], 100 / 3)
            self.assertEqual(metrics["exact_total_count_images"], 2)
            self.assertEqual(set(metrics["failed_images"]), {"swapped.jpg", "extra_unknown.jpg"})


if __name__ == "__main__":
    unittest.main()
