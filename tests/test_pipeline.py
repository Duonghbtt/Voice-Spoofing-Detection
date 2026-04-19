from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from evaluate import evaluate_checkpoint
from src.models import build_model, canonicalize_model_name
from src.utils.dataset import (
    SpoofDataset,
    build_2019_samples,
    build_2021_samples,
    resolve_data_root,
    resolve_external_labels_path,
    resolve_feature_directory,
)
from src.utils.runtime import create_spoof_dataloader, default_batch_size, resolve_num_workers, run_evaluation_loop
from train import ExperimentConfig, _build_signature, build_experiment_config, checkpoint_paths_for_config, train_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_protocol(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for utt_id, label in rows:
            handle.write(f"LA_0001 {utt_id} - - {label}\n")


def _make_feature(utt_id: str, feature_dim: int, frames: int) -> np.ndarray:
    values = np.linspace(0.1, 1.0, num=feature_dim * frames, dtype=np.float32)
    return values.reshape(feature_dim, frames) + (hash(utt_id) % 13) * 0.01


def _feature_split_dir(root: Path, feature_name: str, split: str) -> Path:
    if feature_name == "spectrogram":
        if split == "eval_2021":
            return root / "features" / "output_spectrogram_2021" / "eval"
        return root / "features" / "output_spectrogram_2019" / split

    bundle_name = "output_npy_2021" if split == "eval_2021" else "output_npy_2019"
    split_name = "eval_2021" if split == "eval_2021" else split
    return root / "features" / bundle_name / f"output_{feature_name}" / split_name


def _labels_path(root: Path, feature_name: str, split: str) -> Path:
    filename = f"labels_{split}.csv" if split != "eval_2021" else "labels_eval_2021.csv"
    split_dir = _feature_split_dir(root, feature_name, split)
    return split_dir.parent / filename


def _populate_feature_bundle(
    root: Path,
    feature_name: str,
    split: str,
    rows: list[tuple[str, str]],
    feature_dim: int,
    frames: int,
) -> None:
    split_dir = _feature_split_dir(root, feature_name, split)
    split_dir.mkdir(parents=True, exist_ok=True)
    for index, (utt_id, _) in enumerate(rows, start=1):
        feature = _make_feature(utt_id, feature_dim=feature_dim, frames=frames + index)
        np.save(split_dir / f"{utt_id}.npy", feature)

    label_path = _labels_path(root, feature_name, split)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label"])
        writer.writeheader()
        for utt_id, label in rows:
            writer.writerow({"filename": utt_id, "label": label})


class PipelineSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "data"
        self.protocol_root = self.root / "protocols"
        self.output_root = self.root / "outputs"

        self.train_rows = [("LA_T_0001", "bonafide"), ("LA_T_0002", "spoof"), ("LA_T_0003", "bonafide"), ("LA_T_0004", "spoof")]
        self.dev_rows = [("LA_D_0001", "bonafide"), ("LA_D_0002", "spoof")]
        self.eval_rows = [("LA_E_0001", "bonafide"), ("LA_E_0002", "spoof")]
        self.eval_2021_rows = [("LA_E_2021_0001", "bonafide"), ("LA_E_2021_0002", "spoof")]

        _write_protocol(self.protocol_root / "ASVspoof2019.LA.cm.train.trn.txt", self.train_rows)
        _write_protocol(self.protocol_root / "ASVspoof2019.LA.cm.dev.trl.txt", self.dev_rows)
        _write_protocol(self.protocol_root / "ASVspoof2019.LA.cm.eval.trl.txt", self.eval_rows)

        for feature_name, feature_dim in (("mfcc", 16), ("lfcc", 16), ("spectrogram", 20)):
            _populate_feature_bundle(self.data_root, feature_name, "train", self.train_rows, feature_dim=feature_dim, frames=48)
            _populate_feature_bundle(self.data_root, feature_name, "dev", self.dev_rows, feature_dim=feature_dim, frames=48)
            _populate_feature_bundle(self.data_root, feature_name, "eval", self.eval_rows, feature_dim=feature_dim, frames=48)
            _populate_feature_bundle(self.data_root, feature_name, "eval_2021", self.eval_2021_rows, feature_dim=feature_dim, frames=48)

        self.eval_2021_labels = _labels_path(self.data_root, "mfcc", "eval_2021")

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
            feature_root=None,
            output_root=str(self.output_root),
            seed=42,
            num_workers=0,
            device="cpu",
            resume=False,
            eval_2021=False,
            eval_2021_labels=None,
        )
        return replace(config, **overrides)

    def _read_metrics_json(self, model_name: str, feature_name: str) -> dict:
        metrics_path = self.output_root / "results" / f"{model_name}_{feature_name}" / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _read_train_log(self, model_name: str, feature_name: str) -> pd.DataFrame:
        train_log_path = self.output_root / "results" / f"{model_name}_{feature_name}" / "train_log.csv"
        return pd.read_csv(train_log_path)

    def _write_metrics_json_stub(
        self,
        model_name: str,
        feature_name: str,
        *,
        metrics_2019,
        metrics_2021,
    ) -> None:
        metrics_path = self.output_root / "results" / f"{model_name}_{feature_name}" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": model_name,
                    "feature": feature_name,
                    "metrics_2019": metrics_2019,
                    "metrics_2021": metrics_2021,
                },
                handle,
            )

    def _run_model_script(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(REPO_ROOT / "scripts" / "run_model_developer.ps1"),
                "-PythonExe",
                sys.executable,
                *extra_args,
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

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

    def test_resolve_feature_directory_matches_local_layout(self) -> None:
        self.assertEqual(
            resolve_feature_directory(feature_name="mfcc", split="train", data_root=self.data_root),
            self.data_root / "features" / "output_npy_2019" / "output_mfcc" / "train",
        )
        self.assertEqual(
            resolve_feature_directory(feature_name="lfcc", split="eval_2021", data_root=self.data_root),
            self.data_root / "features" / "output_npy_2021" / "output_lfcc" / "eval_2021",
        )
        self.assertEqual(
            resolve_feature_directory(feature_name="spectrogram", split="eval_2021", data_root=self.data_root),
            self.data_root / "features" / "output_spectrogram_2021" / "eval",
        )

    def test_resolve_external_labels_path_uses_feature_specific_layout(self) -> None:
        self.assertEqual(
            resolve_external_labels_path(feature_name="mfcc", data_root=self.data_root),
            self.data_root / "features" / "output_npy_2021" / "output_mfcc" / "labels_eval_2021.csv",
        )
        self.assertEqual(
            resolve_external_labels_path(feature_name="spectrogram", data_root=self.data_root),
            self.data_root / "features" / "output_spectrogram_2021" / "labels_eval_2021.csv",
        )

    def test_build_2019_samples_uses_feature_bundle_csv_without_protocol_warning(self) -> None:
        (self.protocol_root / "ASVspoof2019.LA.cm.train.trn.txt").unlink()
        with self.assertNoLogs("src.utils.dataset", level="WARNING"):
            samples = build_2019_samples(
                feature_name="mfcc",
                split="train",
                data_root=self.data_root,
                protocol_root=str(self.protocol_root / "missing"),
            )

        self.assertEqual(len(samples), len(self.train_rows))

    def test_build_2019_samples_falls_back_to_protocol_when_csv_invalid(self) -> None:
        label_path = _labels_path(self.data_root, "mfcc", "train")
        with label_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "label"])
            writer.writeheader()
            for utt_id, label in self.train_rows:
                writer.writerow({"filename": f"{utt_id}_missing", "label": label})

        with self.assertLogs("src.utils.dataset", level="WARNING") as captured:
            samples = build_2019_samples(
                feature_name="mfcc",
                split="train",
                data_root=self.data_root,
                protocol_root=self.protocol_root,
            )

        self.assertEqual(len(samples), len(self.train_rows))
        self.assertEqual(len(captured.output), 1)
        self.assertIn("CSV khong hop le, fallback sang protocol", "\n".join(captured.output))

    def test_build_2019_samples_fails_when_no_valid_label_source(self) -> None:
        label_path = _labels_path(self.data_root, "mfcc", "train")
        label_path.unlink()

        with self.assertRaisesRegex(RuntimeError, "Khong co nguon nhan hop le cho split='train'"):
            build_2019_samples(
                feature_name="mfcc",
                split="train",
                data_root=self.data_root,
                protocol_root=str(self.protocol_root / "missing"),
            )

    def test_resolve_data_root_rejects_bundle_like_paths(self) -> None:
        invalid_roots = (
            self.data_root / "features",
            self.data_root / "features" / "output_npy_2019",
            self.data_root / "features" / "output_spectrogram_2021",
        )

        for invalid_root in invalid_roots:
            with self.subTest(data_root=invalid_root):
                with self.assertRaisesRegex(ValueError, "parent data directory"):
                    resolve_data_root(invalid_root)

    def test_model_forward_shapes_and_alias(self) -> None:
        batch = torch.randn(2, 1, 16, 32)
        self.assertEqual(canonicalize_model_name("resnet"), "resnet18")
        for model_name in ("cnn", "resnet", "resnet18", "lcnn"):
            model = build_model(model_name)
            logits = model(batch)
            self.assertEqual(tuple(logits.shape), (2, 2))

    def test_default_batch_size_matrix(self) -> None:
        self.assertEqual(default_batch_size("cnn", "mfcc", device_name="cpu"), 64)
        self.assertEqual(default_batch_size("cnn", "spectrogram", device_name="cpu"), 32)
        self.assertEqual(default_batch_size("lcnn", "spectrogram", device_name="cpu"), 16)
        self.assertEqual(default_batch_size("resnet18", "spectrogram", device_name="cpu"), 8)

    def test_default_batch_size_matrix_for_4gb_cuda(self) -> None:
        with patch("src.utils.runtime._is_windows_host", return_value=False), patch(
            "src.utils.runtime._cuda_total_memory_gb",
            return_value=4.0,
        ):
            self.assertEqual(default_batch_size("cnn", "mfcc", device_name=torch.device("cuda")), 256)
            self.assertEqual(default_batch_size("cnn", "spectrogram", device_name=torch.device("cuda")), 160)
            self.assertEqual(default_batch_size("lcnn", "mfcc", device_name=torch.device("cuda")), 192)
            self.assertEqual(default_batch_size("lcnn", "spectrogram", device_name=torch.device("cuda")), 128)
            self.assertEqual(default_batch_size("resnet18", "mfcc", device_name=torch.device("cuda")), 256)
            self.assertEqual(default_batch_size("resnet18", "spectrogram", device_name=torch.device("cuda")), 128)

    def test_default_batch_size_matrix_for_4gb_cuda_on_windows(self) -> None:
        with patch("src.utils.runtime._is_windows_host", return_value=True), patch(
            "src.utils.runtime._cuda_total_memory_gb",
            return_value=4.0,
        ):
            self.assertEqual(default_batch_size("cnn", "mfcc", device_name=torch.device("cuda")), 128)
            self.assertEqual(default_batch_size("cnn", "spectrogram", device_name=torch.device("cuda")), 80)
            self.assertEqual(default_batch_size("lcnn", "mfcc", device_name=torch.device("cuda")), 96)
            self.assertEqual(default_batch_size("lcnn", "spectrogram", device_name=torch.device("cuda")), 64)
            self.assertEqual(default_batch_size("resnet18", "mfcc", device_name=torch.device("cuda")), 128)
            self.assertEqual(default_batch_size("resnet18", "spectrogram", device_name=torch.device("cuda")), 64)

    def test_build_experiment_config_defaults_to_normalized_lcnn_mfcc(self) -> None:
        args = argparse.Namespace(
            profile=None,
            model="lcnn",
            feature="mfcc",
            epochs=None,
            batch_size=2,
            lr=None,
            scheduler=None,
            step_size=None,
            gamma=None,
            target_frames=128,
            early_stopping_patience=None,
            data_root=str(self.data_root),
            protocol_root=str(self.protocol_root),
            feature_root=None,
            output_root=str(self.output_root),
            seed=42,
            num_workers=0,
            device="cpu",
            resume=False,
            eval_2021=False,
            eval_2021_labels=None,
            normalize=None,
            specaugment=None,
            amp=False,
        )

        config = build_experiment_config(args)
        self.assertTrue(config.normalize)
        self.assertFalse(config.specaugment)

        args.normalize = False
        explicit_config = build_experiment_config(args)
        self.assertFalse(explicit_config.normalize)

    def test_default_batch_size_falls_back_when_free_cuda_memory_is_low(self) -> None:
        with patch("src.utils.runtime._cuda_total_memory_gb", return_value=4.0), patch(
            "src.utils.runtime._cuda_free_memory_gb",
            return_value=1.5,
        ):
            self.assertEqual(default_batch_size("cnn", "mfcc", device_name=torch.device("cuda")), 64)
            self.assertEqual(default_batch_size("cnn", "spectrogram", device_name=torch.device("cuda")), 32)
            self.assertEqual(default_batch_size("lcnn", "spectrogram", device_name=torch.device("cuda")), 16)
            self.assertEqual(default_batch_size("resnet18", "spectrogram", device_name=torch.device("cuda")), 8)

    def test_resolve_num_workers_defaults_to_zero_on_windows(self) -> None:
        with patch("src.utils.runtime._is_windows_host", return_value=True):
            self.assertEqual(resolve_num_workers(None), 0)

    def test_resolve_num_workers_warns_but_honors_explicit_values_on_windows(self) -> None:
        with patch("src.utils.runtime._is_windows_host", return_value=True):
            with self.assertWarnsRegex(UserWarning, "explicit num_workers=8"):
                self.assertEqual(resolve_num_workers(8), 8)
            self.assertEqual(resolve_num_workers(1), 1)

    def test_create_spoof_dataloader_omits_prefetch_knobs_on_windows(self) -> None:
        with patch("src.utils.runtime._is_windows_host", return_value=True), patch("src.utils.runtime.DataLoader") as mock_loader:
            create_spoof_dataloader(
                feature_name="mfcc",
                split="train",
                target_frames=32,
                batch_size=2,
                data_root=str(self.data_root),
                protocol_root=str(self.protocol_root),
                training=True,
                num_workers=2,
                device=torch.device("cuda"),
            )

        kwargs = mock_loader.call_args.kwargs
        self.assertEqual(kwargs["num_workers"], 2)
        self.assertFalse(kwargs["pin_memory"])
        self.assertNotIn("persistent_workers", kwargs)
        self.assertNotIn("prefetch_factor", kwargs)

    def test_create_spoof_dataloader_keeps_prefetch_knobs_off_windows(self) -> None:
        with patch("src.utils.runtime._is_windows_host", return_value=False), patch("src.utils.runtime.DataLoader") as mock_loader:
            create_spoof_dataloader(
                feature_name="mfcc",
                split="train",
                target_frames=32,
                batch_size=2,
                data_root=str(self.data_root),
                protocol_root=str(self.protocol_root),
                training=True,
                num_workers=2,
                device=torch.device("cuda"),
            )

        kwargs = mock_loader.call_args.kwargs
        self.assertEqual(kwargs["num_workers"], 2)
        self.assertTrue(kwargs["pin_memory"])
        self.assertTrue(kwargs["persistent_workers"])
        self.assertEqual(kwargs["prefetch_factor"], 4)

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

    def test_resume_recreates_best_checkpoint_and_uses_new_checkpoint_dir(self) -> None:
        initial_config = self._make_config(profile="optimized", model="lcnn", feature="lfcc", epochs=1, normalize=True, specaugment=True, amp=True)
        resumed_config = replace(initial_config, epochs=2, resume=True)

        initial_result = train_experiment(initial_config)
        checkpoint_paths = checkpoint_paths_for_config(initial_config)

        self.assertIn("lcnn_lfcc", initial_result["checkpoint_path"])
        self.assertTrue(checkpoint_paths["best"].exists())

        checkpoint_paths["best"].unlink()
        resumed_result = train_experiment(resumed_config)

        self.assertTrue(Path(resumed_result["checkpoint_path"]).exists())
        last_checkpoint = torch.load(checkpoint_paths["last"], map_location="cpu")
        best_checkpoint = torch.load(checkpoint_paths["best"], map_location="cpu")
        self.assertTrue(bool(last_checkpoint["training_complete"]))
        self.assertTrue(bool(best_checkpoint["training_complete"]))
        self.assertEqual(int(last_checkpoint["epoch"]), 2)

    def test_resume_ignores_legacy_batch_size_metadata(self) -> None:
        initial_config = self._make_config(model="cnn", feature="lfcc", epochs=1, batch_size=2)
        resumed_config = replace(initial_config, epochs=2, resume=True, batch_size=128)

        train_experiment(initial_config)
        checkpoint_paths = checkpoint_paths_for_config(initial_config)
        last_checkpoint = torch.load(checkpoint_paths["last"], map_location="cpu")
        last_checkpoint["batch_size"] = 256
        last_checkpoint["resume_signature"]["raw"]["batch_size"] = 256
        torch.save(last_checkpoint, checkpoint_paths["last"])

        resumed_result = train_experiment(resumed_config)

        resumed_checkpoint = torch.load(checkpoint_paths["last"], map_location="cpu")
        self.assertTrue(Path(resumed_result["checkpoint_path"]).exists())
        self.assertEqual(int(resumed_checkpoint["epoch"]), 2)
        self.assertEqual(int(resumed_checkpoint["batch_size"]), 128)

    def test_resume_lcnn_mfcc_restarts_from_unnormalized_checkpoint(self) -> None:
        initial_config = self._make_config(profile="custom", model="lcnn", feature="mfcc", epochs=1, normalize=False, amp=False)
        resumed_config = replace(initial_config, epochs=2, normalize=True, resume=True)

        train_experiment(initial_config)
        resumed_result = train_experiment(resumed_config)

        checkpoint_paths = checkpoint_paths_for_config(resumed_config)
        resumed_checkpoint = torch.load(checkpoint_paths["last"], map_location="cpu")
        self.assertTrue(Path(resumed_result["checkpoint_path"]).exists())
        self.assertTrue(bool(resumed_checkpoint["normalize"]))
        self.assertEqual(int(resumed_checkpoint["epoch"]), 2)

    def test_train_experiment_writes_train_log_after_first_epoch(self) -> None:
        config = self._make_config(model="cnn", feature="mfcc", epochs=1, batch_size=2)

        train_experiment(config)

        train_log = self._read_train_log("cnn", "mfcc")
        self.assertEqual(
            train_log.columns.tolist(),
            ["profile", "model", "feature", "epoch", "train_loss", "val_loss", "val_accuracy", "val_eer", "lr"],
        )
        self.assertEqual(len(train_log), 1)
        self.assertEqual(train_log["epoch"].tolist(), [1])

    def test_train_experiment_keeps_partial_train_log_when_interrupted(self) -> None:
        config = self._make_config(model="cnn", feature="mfcc", epochs=2, batch_size=2)
        train_log_path = self.output_root / "results" / "cnn_mfcc" / "train_log.csv"
        evaluation_calls = {"count": 0}

        def interrupted_evaluation(*args, **kwargs):
            evaluation_calls["count"] += 1
            if evaluation_calls["count"] == 2:
                raise RuntimeError("Simulated interruption during validation")
            return run_evaluation_loop(*args, **kwargs)

        with patch("train.run_evaluation_loop", side_effect=interrupted_evaluation):
            with self.assertRaisesRegex(RuntimeError, "Simulated interruption during validation"):
                train_experiment(config)

        self.assertTrue(train_log_path.exists())
        train_log = self._read_train_log("cnn", "mfcc")
        self.assertEqual(train_log["epoch"].tolist(), [1])
        self.assertEqual(len(train_log), 1)

    def test_run_evaluation_loop_raises_on_nonfinite_logits(self) -> None:
        class NonFiniteModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((x.size(0), 2), float("nan"), dtype=x.dtype, device=x.device)

        dataset = [
            (torch.zeros(1, 16, 32), torch.tensor(0, dtype=torch.long), "utt_a"),
            (torch.zeros(1, 16, 32), torch.tensor(1, dtype=torch.long), "utt_b"),
        ]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        with self.assertRaisesRegex(RuntimeError, "Non-finite logits detected during evaluation"):
            run_evaluation_loop(
                model=NonFiniteModel(),
                dataloader=dataloader,
                device=torch.device("cpu"),
                criterion=None,
                use_amp=False,
            )

    def test_evaluate_checkpoint_writes_metrics_json_for_2019_only(self) -> None:
        config = self._make_config(model="cnn", feature="mfcc", epochs=1, batch_size=2)
        result = train_experiment(config)

        evaluation_result = evaluate_checkpoint(
            checkpoint_path=result["checkpoint_path"],
            data_root=str(self.data_root),
            protocol_root=str(self.protocol_root),
            output_root=str(self.output_root),
            batch_size=2,
            num_workers=0,
            device="cpu",
            run_2019=True,
            run_2021=False,
        )

        metrics_json = self._read_metrics_json("cnn", "mfcc")
        self.assertEqual(Path(evaluation_result["metrics_json_path"]).name, "metrics.json")
        self.assertEqual(metrics_json["model"], "cnn")
        self.assertEqual(metrics_json["feature"], "mfcc")
        self.assertIsNone(metrics_json["metrics_2021"])
        self.assertEqual(metrics_json["metrics_2019"]["num_samples"], len(self.eval_rows))
        self.assertEqual(metrics_json["config"]["batch_size"], 2)
        self.assertEqual(metrics_json["config"]["num_workers"], 0)
        self.assertEqual(metrics_json["config"]["device"], "cpu")
        self.assertIsNotNone(metrics_json["best_epoch"])
        self.assertIsInstance(datetime.fromisoformat(metrics_json["generated_at"]), datetime)

    def test_metrics_json_best_epoch_is_null_when_checkpoint_has_no_epoch(self) -> None:
        config = self._make_config(model="cnn", feature="lfcc", epochs=1, batch_size=2)
        result = train_experiment(config)
        checkpoint_path = Path(result["checkpoint_path"])
        checkpoint_bundle = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_bundle.pop("best_epoch", None)
        checkpoint_bundle.pop("epoch", None)
        torch.save(checkpoint_bundle, checkpoint_path)

        evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
            data_root=str(self.data_root),
            protocol_root=str(self.protocol_root),
            output_root=str(self.output_root),
            batch_size=2,
            num_workers=0,
            device="cpu",
            run_2019=True,
            run_2021=False,
        )

        metrics_json = self._read_metrics_json("cnn", "lfcc")
        self.assertIsNone(metrics_json["best_epoch"])

    def test_evaluate_checkpoint_uses_current_auto_batch_size_instead_of_checkpoint_batch_size(self) -> None:
        config = self._make_config(model="cnn", feature="lfcc", epochs=1, batch_size=2)
        result = train_experiment(config)
        checkpoint_path = Path(result["checkpoint_path"])
        checkpoint_bundle = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_bundle["batch_size"] = 256
        torch.save(checkpoint_bundle, checkpoint_path)

        with patch("evaluate.default_batch_size", return_value=128):
            evaluate_checkpoint(
                checkpoint_path=str(checkpoint_path),
                data_root=str(self.data_root),
                protocol_root=str(self.protocol_root),
                output_root=str(self.output_root),
                batch_size=None,
                num_workers=0,
                device="cpu",
                run_2019=True,
                run_2021=False,
            )

        metrics_json = self._read_metrics_json("cnn", "lfcc")
        self.assertEqual(metrics_json["config"]["batch_size"], 128)

    def test_train_cli_runs_post_train_eval_2021(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "train.py",
                "--model",
                "cnn",
                "--feature",
                "mfcc",
                "--epochs",
                "1",
                "--batch_size",
                "2",
                "--num_workers",
                "0",
                "--eval_2021",
                "--data_root",
                str(self.data_root),
                "--protocol_root",
                str(self.protocol_root),
                "--output_root",
                str(self.output_root),
                "--num_workers",
                "0",
                "--device",
                "cpu",
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        self.assertTrue((self.output_root / "checkpoints" / "cnn_mfcc" / "best.ckpt").exists())
        self.assertTrue((self.output_root / "results" / "cnn_mfcc" / "train_log.csv").exists())
        self.assertTrue((self.output_root / "results" / "cnn_mfcc" / "results_2019.csv").exists())
        self.assertTrue((self.output_root / "results" / "cnn_mfcc" / "results_2021.csv").exists())
        self.assertTrue((self.output_root / "results" / "cnn_mfcc" / "predictions_2021.csv").exists())
        self.assertTrue((self.output_root / "figures" / "cnn_mfcc" / "confusion_matrix_2019.png").exists())
        metrics_json = self._read_metrics_json("cnn", "mfcc")
        self.assertEqual(metrics_json["metrics_2019"]["num_samples"], len(self.eval_rows))
        self.assertEqual(metrics_json["metrics_2021"]["num_samples"], len(self.eval_2021_rows))
        self.assertIn("generated_at", metrics_json)
        self.assertIn("best_epoch", metrics_json)

    def test_evaluate_cli_supports_spectrogram_layout(self) -> None:
        config = self._make_config(model="cnn", feature="spectrogram", profile="custom", epochs=1, batch_size=2)
        result = train_experiment(config)

        subprocess.run(
            [
                sys.executable,
                "evaluate.py",
                "--checkpoint",
                result["checkpoint_path"],
                "--eval_2019",
                "--eval_2021",
                "--data_root",
                str(self.data_root),
                "--protocol_root",
                str(self.protocol_root),
                "--output_root",
                str(self.output_root),
                "--num_workers",
                "0",
                "--device",
                "cpu",
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        self.assertTrue((self.output_root / "results" / "cnn_spectrogram" / "results_2021.csv").exists())
        self.assertTrue((self.output_root / "results" / "cnn_spectrogram" / "eer_comparison.csv").exists())
        metrics_json = self._read_metrics_json("cnn", "spectrogram")
        self.assertEqual(metrics_json["metrics_2021"]["num_samples"], len(self.eval_2021_rows))

    def test_run_model_script_all_combinations_skips_completed_checkpoints(self) -> None:
        combinations = (
            ("cnn", "mfcc"),
            ("cnn", "lfcc"),
            ("cnn", "spectrogram"),
            ("lcnn", "mfcc"),
            ("lcnn", "lfcc"),
            ("lcnn", "spectrogram"),
            ("resnet18", "mfcc"),
            ("resnet18", "lfcc"),
            ("resnet18", "spectrogram"),
        )
        for model_name, feature_name in combinations:
            checkpoint_dir = self.output_root / "checkpoints" / f"{model_name}_{feature_name}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"training_complete": True}, checkpoint_dir / "best.ckpt")

        self._run_model_script(
            "-DataRoot",
            str(self.root / "missing-data"),
            "-OutputRoot",
            str(self.output_root),
            "-AllCombinations",
            "-SkipEval2019",
            "-SkipEval2021",
        )

    def test_run_model_script_skips_eval_when_metrics_complete(self) -> None:
        self._write_metrics_json_stub(
            "cnn",
            "mfcc",
            metrics_2019={"eer": 0.1},
            metrics_2021={"eer": 0.2},
        )

        result = self._run_model_script(
            "-DataRoot",
            str(self.data_root),
            "-OutputRoot",
            str(self.output_root),
            "-Model",
            "cnn",
            "-Feature",
            "mfcc",
            "-SkipTrain",
            "-Eval2021",
        )

        self.assertIn("[SKIP EVAL] cnn_mfcc da co metrics day du", result.stdout)

    def test_run_model_script_lets_python_choose_runtime_defaults(self) -> None:
        config = self._make_config(model="cnn", feature="mfcc", epochs=1, batch_size=2)
        result = train_experiment(config)
        self._write_metrics_json_stub(
            "cnn",
            "mfcc",
            metrics_2019={"eer": 0.1},
            metrics_2021=None,
        )

        script_result = self._run_model_script(
            "-DataRoot",
            str(self.data_root),
            "-ProtocolRoot",
            str(self.protocol_root),
            "-OutputRoot",
            str(self.output_root),
            "-Model",
            "cnn",
            "-Feature",
            "mfcc",
            "-SkipTrain",
            "-Eval2021",
        )

        self.assertTrue(Path(result["checkpoint_path"]).exists())
        self.assertIn("python", script_result.stdout)
        self.assertIn("evaluate.py", script_result.stdout)
        self.assertNotIn("--batch_size", script_result.stdout)
        self.assertNotIn("--num_workers", script_result.stdout)

    def test_run_model_script_runs_eval_when_metrics_incomplete(self) -> None:
        config = self._make_config(model="cnn", feature="mfcc", epochs=1, batch_size=2)
        result = train_experiment(config)
        self._write_metrics_json_stub(
            "cnn",
            "mfcc",
            metrics_2019={"eer": 0.1},
            metrics_2021=None,
        )

        script_result = self._run_model_script(
            "-DataRoot",
            str(self.data_root),
            "-ProtocolRoot",
            str(self.protocol_root),
            "-OutputRoot",
            str(self.output_root),
            "-Model",
            "cnn",
            "-Feature",
            "mfcc",
            "-SkipTrain",
            "-Eval2021",
            "-BatchSize",
            "2",
            "-NumWorkers",
            "0",
        )

        self.assertTrue(Path(result["checkpoint_path"]).exists())
        self.assertIn("[RUN EVAL] cnn_mfcc chua co metrics day du -> chay evaluate", script_result.stdout)
        self.assertIn("--batch_size 2", script_result.stdout)
        self.assertIn("--num_workers 0", script_result.stdout)
        metrics_json = self._read_metrics_json("cnn", "mfcc")
        self.assertIsNotNone(metrics_json["metrics_2019"])
        self.assertIsNotNone(metrics_json["metrics_2021"])

    def test_run_model_script_runs_eval_when_metrics_json_is_malformed(self) -> None:
        config = self._make_config(model="cnn", feature="mfcc", epochs=1, batch_size=2)
        result = train_experiment(config)
        metrics_path = self.output_root / "results" / "cnn_mfcc" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("{invalid json", encoding="utf-8")

        script_result = self._run_model_script(
            "-DataRoot",
            str(self.data_root),
            "-ProtocolRoot",
            str(self.protocol_root),
            "-OutputRoot",
            str(self.output_root),
            "-Model",
            "cnn",
            "-Feature",
            "mfcc",
            "-SkipTrain",
            "-Eval2021",
            "-BatchSize",
            "2",
            "-NumWorkers",
            "0",
        )

        self.assertTrue(Path(result["checkpoint_path"]).exists())
        self.assertIn("[RUN EVAL] cnn_mfcc chua co metrics day du -> chay evaluate", script_result.stdout)
        metrics_json = self._read_metrics_json("cnn", "mfcc")
        self.assertIsNotNone(metrics_json["metrics_2019"])
        self.assertIsNotNone(metrics_json["metrics_2021"])

    def test_build_2021_samples_warns_and_fails_cleanly_when_labels_missing(self) -> None:
        label_path = _labels_path(self.data_root, "spectrogram", "eval_2021")
        label_path.unlink()

        with self.assertLogs("src.utils.dataset", level="WARNING") as captured:
            with self.assertRaisesRegex(RuntimeError, "Khong co sample co nhan hop le"):
                build_2021_samples(feature_name="spectrogram", data_root=self.data_root)

        self.assertIn("Khong tim thay file nhan CSV", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
