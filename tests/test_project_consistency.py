"""Kiểm tra tính nhất quán giữa config, dataset và training artifacts."""

from __future__ import annotations

import csv
import json
import unittest

import yaml

from project_config import CONFIG, project_path


class ProjectConsistencyTests(unittest.TestCase):
    def test_dataset_yaml_matches_project_config(self) -> None:
        data_yaml = project_path(CONFIG["dataset"]["yaml"])
        with data_yaml.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        self.assertEqual(data["nc"], len(CONFIG["dataset"]["classes"]))
        self.assertEqual(data["names"], CONFIG["dataset"]["classes"])
        self.assertEqual(data["train"], "train/images")
        self.assertEqual(data["val"], "valid/images")
        self.assertEqual(data["test"], "test/images")

    def test_training_history_matches_manifest(self) -> None:
        run_dir = project_path(CONFIG["training"]["artifact_dir"])
        with (run_dir / "results.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), CONFIG["training"]["epochs"])
        best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
        self.assertEqual(int(float(best["epoch"])), 47)

    def test_test_metrics_and_model_exist(self) -> None:
        run_dir = project_path(CONFIG["training"]["artifact_dir"])
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertTrue((run_dir / "weights" / "best.pt").is_file())
        self.assertEqual(metrics["split"], "Test")
        self.assertEqual(metrics["images"], CONFIG["dataset"]["image_counts"]["test"])
        self.assertEqual(
            [item["name"] for item in metrics["per_class"]],
            CONFIG["dataset"]["classes"],
        )

    def test_inference_thresholds_are_two_stage(self) -> None:
        inference = CONFIG["inference"]
        self.assertLess(inference["candidate_confidence"], inference["confirmed_confidence"])
        self.assertEqual(inference["imgsz"], CONFIG["training"]["imgsz"])


if __name__ == "__main__":
    unittest.main()
