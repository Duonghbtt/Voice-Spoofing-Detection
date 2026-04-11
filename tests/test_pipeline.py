from __future__ import annotations

import csv
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import build_model
from src.utils.dataset import SpoofDataset, build_2019_samples, resolve_feature_directory
from train import ExperimentConfig, _build_signature, checkpoint_paths_for_config, train_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_protocol(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for utt_id, label in rows:
            handle.write(f"LA_0001 {utt_id} - - {label}\n")


def _make_feature(utt_id: str, feature_dim: int, frames: int, scale: float = 1.0) -> np.ndarray:
    values = np.linspace(0.1, 1.0, num=feature_dim * frames, dtype=np.float32)
    return (values.reshape(feature_dim, frames) * scale) + (hash(utt_id) % 13) * 0.01


def _populate_feature_dir(root: Path, feature_name: str, split: str, rows: list[tuple[str, str]], feature_dim: int, frames: int):
    split_dir = root / "features" / feature_name / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for index, (utt_id, _) in enumerate(rows, start=1):
        feature = _make_feature(utt_id, feature_dim=feature_dim, frames=frames + index)
        np.save(split_dir / f"{utt_id}.npy", feature)


def _populate_bundle_feature_dir(
    root: Path,
    bundle_name: str,
    feature_name: str,
    split: str,
    rows: list[tuple[str, str]],
    feature_dim: int,
    frames: int,
) -> None:
    split_dir = root / bundle_name / f"output_{feature_name}" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for index, (utt_id, _) in enumerate(rows, start=1):
        feature = _make_feature(utt_id, feature_dim=feature_dim, frames=frames + index)
        np.save(split_dir / f"{utt_id}.npy", feature)


class PipelineSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "data"
        self.protocol_root = self.root / "protocols"
        self.output_root = self.root / "outputs"

        train_rows = [("LA_T_0001", "bonafide"), ("LA_T_0002", "spoof"), ("LA_T_0003", "bonafide"), ("LA_T_0004", "spoof")]
        dev_rows = [("LA_D_0001", "bonafide"), ("LA_D_0002", "spoof")]
        eval_rows = [("LA_E_0001", "bonafide"), ("LA_E_0002", "spoof")]
        eval_2021_rows = [("LA_E_2021_0001", "bonafide"), ("LA_E_2021_0002", "spoof")]

        _write_protocol(self.protocol_root / "ASVspoof2019.LA.cm.train.trn.txt", train_rows)
        _write_protocol(self.protocol_root / "ASVspoof2019.LA.cm.dev.trl.txt", dev_rows)
        _write_protocol(self.protocol_root / "ASVspoof2019.LA.cm.eval.trl.txt", eval_rows)

        for feature_name in ("mfcc", "lfcc"):
            _populate_feature_dir(self.data_root, feature_name, "train", train_rows, feature_dim=16, frames=48)
            _populate_feature_dir(self.data_root, feature_name, "dev", dev_rows, feature_dim=16, frames=48)
            _populate_feature_dir(self.data_root, feature_name, "eval", eval_rows, feature_dim=16, frames=48)
            _populate_feature_dir(self.data_root, feature_name, "eval_2021", eval_2021_rows, feature_dim=16, frames=48)

        labels_path = self.root / "labels_2021.csv"
        with labels_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["utt_id", "label"])
            writer.writeheader()
            for utt_id, label in eval_2021_rows:
                writer.writerow({"utt_id": utt_id, "label": label})
        self.eval_2021_labels = labels_path

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _make_config(self, **overrides) -> ExperimentConfig:
        config = ExperimentConfig(
            profile="baseline",
            model="cnn",
            feature="mfcc",
            epochs=1,
            batch_size=2,
            lr=1e-3,
            scheduler="step",
            step_size=15,
            gamma=0.5,
            target_frames=32,
            normalize=False,
            specaugment=False,
            amp=False,
            early_stopping_patience=None,
            data_root=str(self.data_root),
            protocol_root=str(self.protocol_root),
            output_root=str(self.output_root),
            seed=42,
            num_workers=0,
            device="cpu",
            resume=False,
        )
        return replace(config, **overrides)

    def test_dataset_padding_normalization_and_specaugment(self) -> None:
        samples = build_2019_samples(
            feature_name="lfcc",
            split="train",
            data_root=self.data_root,
            protocol_root=self.protocol_root,
        )

        baseline_dataset = SpoofDataset(
            samples=samples,
            target_frames=32,
            training=False,
            normalize=False,
            apply_lfcc_specaugment=False,
            feature_name="lfcc",
        )
        optimized_dataset = SpoofDataset(
            samples=samples,
            target_frames=32,
            training=True,
            normalize=True,
            apply_lfcc_specaugment=True,
            feature_name="lfcc",
        )

        baseline_tensor, _, _ = baseline_dataset[0]
        self.assertEqual(tuple(baseline_tensor.shape), (1, 16, 32))

        random.seed(7)
        optimized_tensor_a, _, _ = optimized_dataset[0]
        random.seed(7)
        optimized_dataset_no_aug = SpoofDataset(
            samples=samples,
            target_frames=32,
            training=True,
            normalize=True,
            apply_lfcc_specaugment=False,
            feature_name="lfcc",
        )
        optimized_tensor_b, _, _ = optimized_dataset_no_aug[0]

        self.assertAlmostEqual(float(optimized_tensor_b.mean()), 0.0, places=4)
        self.assertFalse(torch.equal(optimized_tensor_a, optimized_tensor_b))

    def test_resolve_feature_directory_supports_raw_output_layout(self) -> None:
        raw_root = self.root / "raw-layout-data"
        train_rows = [("LA_T_1001", "bonafide"), ("LA_T_1002", "spoof")]
        eval_2021_rows = [("LA_E_2021_1001", "bonafide"), ("LA_E_2021_1002", "spoof")]

        _populate_bundle_feature_dir(raw_root / "raw", "output_npy_2019", "mfcc", "train", train_rows, feature_dim=16, frames=48)
        _populate_bundle_feature_dir(
            raw_root / "raw",
            "output_npy_2021",
            "mfcc",
            "eval_2021",
            eval_2021_rows,
            feature_dim=16,
            frames=48,
        )

        train_dir = resolve_feature_directory(feature_name="mfcc", split="train", data_root=raw_root)
        eval_2021_dir = resolve_feature_directory(feature_name="mfcc", split="eval_2021", data_root=raw_root)

        self.assertEqual(train_dir, raw_root / "raw" / "output_npy_2019" / "output_mfcc" / "train")
        self.assertEqual(eval_2021_dir, raw_root / "raw" / "output_npy_2021" / "output_mfcc" / "eval_2021")

    def test_model_forward_shapes(self) -> None:
        batch = torch.randn(2, 1, 16, 32)
        for model_name in ("cnn", "resnet", "lcnn"):
            model = build_model(model_name)
            logits = model(batch)
            self.assertEqual(tuple(logits.shape), (2, 2))

    def test_signature_canonicalization_ignores_key_order_and_default_omissions(self) -> None:
        defaults = {
            "in_channels": 1,
            "num_classes": 2,
            "class_mapping": {"bonafide": 0, "spoof": 1},
            "target_frames": 128,
            "model_kwargs": {"dropout": 0.3},
        }
        signature_a = _build_signature(
            raw_payload={
                "model_name": "cnn",
                "feature_name": "mfcc",
                "target_frames": 128,
                "class_mapping": {"spoof": 1, "bonafide": 0},
                "model_kwargs": {"dropout": 0.3},
            },
            defaults=defaults,
        )
        signature_b = _build_signature(
            raw_payload={
                "feature_name": "mfcc",
                "model_name": "cnn",
                "class_mapping": {"bonafide": 0, "spoof": 1},
            },
            defaults=defaults,
        )

        self.assertEqual(signature_a["canonical_json"], signature_b["canonical_json"])
        self.assertEqual(signature_a["hash"], signature_b["hash"])

    def test_resume_recreates_best_checkpoint_and_keeps_resume_state(self) -> None:
        initial_config = self._make_config(profile="optimized", model="lcnn", feature="lfcc", epochs=1, normalize=True, specaugment=True, amp=True)
        resumed_config = replace(initial_config, epochs=2, resume=True)

        initial_result = train_experiment(initial_config)
        checkpoint_paths = checkpoint_paths_for_config(initial_config)
        self.assertTrue(Path(initial_result["checkpoint_path"]).exists())
        self.assertTrue(checkpoint_paths["last"].exists())
        self.assertTrue(checkpoint_paths["best"].exists())

        checkpoint_paths["best"].unlink()
        resumed_result = train_experiment(resumed_config)

        self.assertTrue(Path(resumed_result["checkpoint_path"]).exists())
        last_checkpoint = torch.load(checkpoint_paths["last"], map_location="cpu")
        best_checkpoint = torch.load(checkpoint_paths["best"], map_location="cpu")

        self.assertTrue(bool(last_checkpoint["training_complete"]))
        self.assertTrue(bool(best_checkpoint["training_complete"]))
        self.assertEqual(int(last_checkpoint["epoch"]), 2)
        self.assertEqual(len(last_checkpoint["history"]["train_loss"]), 2)
        self.assertIn("optimizer_state_dict", last_checkpoint)
        self.assertIn("scheduler_state_dict", last_checkpoint)
        self.assertIn("best_state_dict", last_checkpoint)
        self.assertIn("architecture_signature", last_checkpoint)
        self.assertIn("resume_signature", last_checkpoint)
        self.assertEqual(last_checkpoint["architecture_signature"]["signature_version"], "v1")
        self.assertEqual(last_checkpoint["resume_signature"]["signature_version"], "v1")

    def test_resume_rejects_architecture_signature_mismatch(self) -> None:
        base_config = self._make_config(epochs=1)
        train_experiment(base_config)

        mismatched_config = replace(base_config, epochs=2, target_frames=64, resume=True)
        with self.assertRaisesRegex(ValueError, "architecture_signature mismatch"):
            train_experiment(mismatched_config)

    def test_run_model_script_skips_completed_best_checkpoints(self) -> None:
        checkpoint_root = self.output_root / "checkpoints"
        (checkpoint_root / "baseline_cnn_mfcc").mkdir(parents=True, exist_ok=True)
        (checkpoint_root / "optimized_lcnn_lfcc").mkdir(parents=True, exist_ok=True)
        torch.save({"training_complete": True}, checkpoint_root / "baseline_cnn_mfcc" / "best.ckpt")
        torch.save({"training_complete": True}, checkpoint_root / "optimized_lcnn_lfcc" / "best.ckpt")

        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(REPO_ROOT / "scripts" / "run_model_developer.ps1"),
                "-DataRoot",
                str(self.root / "missing-data"),
                "-OutputRoot",
                str(self.output_root),
                "-SkipEval2019",
                "-SkipEval2021",
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    def test_train_and_eval_smoke(self) -> None:
        baseline_config = self._make_config()
        optimized_config = self._make_config(
            profile="optimized",
            model="lcnn",
            feature="lfcc",
            scheduler="cosine",
            normalize=True,
            specaugment=True,
            amp=True,
            early_stopping_patience=2,
        )

        baseline_result = train_experiment(baseline_config)
        optimized_result = train_experiment(optimized_config)

        self.assertTrue(Path(baseline_result["checkpoint_path"]).exists())
        self.assertTrue(Path(optimized_result["checkpoint_path"]).exists())

        train_log = pd.read_csv(self.output_root / "results" / "train_log.csv")
        self.assertEqual(set(train_log["profile"]), {"baseline", "optimized"})

        subprocess.run(
            [
                sys.executable,
                "evaluate.py",
                "--checkpoint",
                baseline_result["checkpoint_path"],
                "--eval_2019",
                "--eval_2021",
                "--eval_2021_labels",
                str(self.eval_2021_labels),
                "--data_root",
                str(self.data_root),
                "--protocol_root",
                str(self.protocol_root),
                "--output_root",
                str(self.output_root),
                "--device",
                "cpu",
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        self.assertTrue((self.output_root / "results" / "results_2019.csv").exists())
        self.assertTrue((self.output_root / "results" / "results_2021.csv").exists())
        self.assertTrue((self.output_root / "results" / "eer_comparison.csv").exists())
        self.assertTrue((self.output_root / "figures" / "baseline_confusion_matrix_2019.png").exists())
        self.assertTrue((self.output_root / "results" / "predictions_baseline_2019.csv").exists())
        self.assertTrue((self.output_root / "results" / "predictions_baseline_2021.csv").exists())


if __name__ == "__main__":
    unittest.main()
