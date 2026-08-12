import json
import tempfile
import unittest
from pathlib import Path

import visualize_stats


class VisualizeStatsTests(unittest.TestCase):
    def setUp(self):
        self.stats = {
            "model_count": 3,
            "downloads_total": 1200,
            "likes_total": 42,
            "followers_total": 7,
            "dimensions": {"license_class": {"open-source": 2, "open-weight": 1}},
        }

    def test_load_and_render_self_contained_dashboard(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stats.json"
            source.write_text(json.dumps(self.stats), encoding="utf-8")
            result = visualize_stats.render_dashboard(
                visualize_stats.load_statistics(source), "Test <view>", 10
            )

        self.assertIn("<!doctype html>", result)
        self.assertIn("Test &lt;view&gt;", result)
        self.assertIn('"open-source": 2', result)
        self.assertNotIn("https://", result)

    def test_rejects_malformed_dimensions(self):
        self.stats["dimensions"] = {"license": {"apache": "many"}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stats.json"
            source.write_text(json.dumps(self.stats), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "license"):
                visualize_stats.load_statistics(source)

    def test_rejects_non_positive_top_value(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            visualize_stats.render_dashboard(self.stats, "Test", 0)


if __name__ == "__main__":
    unittest.main()
