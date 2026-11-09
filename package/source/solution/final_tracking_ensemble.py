"""Fit and deploy the fixed external tracking-only HGB/two-TCN ensemble.

This entry point never opens target labels or runs a scoring/selection loop.
Use ``fit`` to create a new model artifact, then ``infer`` to create a new CSV.
"""

from __future__ import annotations

# Torch must initialize before sklearn-bearing imports on the Windows runtime.
import torch

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits

from solution.ensemble_artifact import (
    ALLOWED_TRAINING_IDS,
    BLEND_WEIGHTS,
    CANONICAL_DURATION_CONFIG,
    SAMPLE_FPS,
    TRAINING_SEEDS,
    FrozenEnsembleArtifact,
    build_ensemble_manifest,
    ensemble_artifact_sha256,
    load_ensemble_artifact,
    save_ensemble_artifact,
    validate_ensemble_artifact,
)
from solution.ensemble_resampling import project_probabilities, resample_tracking
from solution.infer_external import prepare_target_features, write_validated_submission
from solution.target_adapter import load_bar_ath_sequence
from solution.temporal import compose_emissions, decode_runs
from solution.tracking_features import valid_runs
from solution.train_external import ModelConfig, _training_arrays, fit_models, predict_probabilities
from solution.train_tcn_external import (
    ExperimentConfig,
    PreparedMatch,
    SequenceConfig,
    fit_fold_normalizer,
    input_cache_contract,
    load_allowed_matches,
    predict_prepared_match,
    prepare_cached_match,
    train_sequence_model,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data/cache/external_model/experiments"
HGB_REFERENCE = REFERENCE_DIR / "hgb_contact_context_segment_balanced_capacity_v5.json"
ENSEMBLE_REFERENCE = REFERENCE_DIR / (
    "high_capacity_contact_tcn_seed_ensemble_v1_a1556e75758d_d42b1498e21c.json"
)
HGB_REFERENCE_SHA256 = "7d7f3738387c4962583275b7b41bad68bc19144867e7944ec074baab0bf19411"
ENSEMBLE_REFERENCE_SHA256 = "486d5a6f274c1e17b2cab4fe9302ec2288ffff90debdd7a380b141de5bed6cb2"
TRAINER_SHA256 = "53d0e9788f46290a645f78e66a96fc55c274c686fcb7458cae91340595cf73ca"
DEFAULT_CACHE = ROOT / "data/cache/external_model/features_contact_v1"
DEFAULT_MODEL = ROOT / "data/cache/external_model/final_tracking_ensemble_v1.joblib"
DEFAULT_OUTPUT = ROOT / "evaluation/results/external_tracking_ensemble_v1.csv"
LETTERS = np.array(("H", "A", "D", "X"), dtype="<U1")
SOURCE_PATHS = (
    "solution/final_tracking_ensemble.py", "solution/ensemble_artifact.py",
    "solution/ensemble_resampling.py", "solution/train_external.py",
    "solution/train_tcn_external.py", "solution/tracking_features.py",
    "solution/contact_features.py", "solution/contact_cache.py",
    "solution/skillcorner_adapter.py", "solution/external_labels.py",
    "solution/temporal.py", "solution/infer_external.py", "solution/model_artifact.py",
    "solution/target_adapter.py", "starter/baseline.py", "starter/load_atd.py",
    "evaluation/possession_metrics.py", "games.py",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_PATHS}


def target_input_hashes(data_dir=None) -> dict:
    """Bind only automatic tracking and its team/slot metadata, never labels."""
    from games import GAME, require

    names = ("atd_meta", "atd_raw")
    paths = require(list(names), GAME, data_dir)
    return {name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
            for name, path in zip(names, paths)}


def hgb_config() -> ModelConfig:
    return ModelConfig(learning_rate=0.08, max_iter=120, max_leaf_nodes=63,
                       min_samples_leaf=30, l2_regularization=1.0,
                       random_state=20261023)


def tcn_config(seed: int) -> ExperimentConfig:
    if seed not in TRAINING_SEEDS:
        raise ValueError("seed is outside the fixed final training configuration")
    return ExperimentConfig(
        seed=seed, epochs=8, batch_size=8, window_frames=1024, stride_frames=512,
        learning_rate=0.001, weight_decay=0.001, transition_radius_frames=10,
        transition_boost=4.0, segment_weighting="sqrt_inverse", segment_weight_cap=8.0,
        sequence=SequenceConfig(96, (1, 2, 4, 8, 16, 32, 64), 0.1),
    )


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validate_reference_reports() -> tuple[dict, dict]:
    """Verify external selection evidence before any training cache is opened."""
    for path, expected in ((HGB_REFERENCE, HGB_REFERENCE_SHA256),
                           (ENSEMBLE_REFERENCE, ENSEMBLE_REFERENCE_SHA256)):
        if sha256(path) != expected:
            raise ValueError("external reference report hash differs: {}".format(path))
    hgb = json.loads(HGB_REFERENCE.read_text(encoding="utf-8"))
    ensemble = json.loads(ENSEMBLE_REFERENCE.read_text(encoding="utf-8"))
    if hgb["model_config"] != asdict(hgb_config()) or hgb["sample_stride"] != 5 \
            or hgb["feature_count"] != 253:
        raise ValueError("accepted HGB settings differ from frozen final settings")
    if ensemble["decision"] != "accept-seed-ensemble" \
            or ensemble["config"] != _json_value(asdict(tcn_config(TRAINING_SEEDS[1]))):
        raise ValueError("accepted TCN settings/decision differ")
    expected_ids = {str(item) for item in ALLOWED_TRAINING_IDS}
    if set(ensemble["provenance"]["input_cache_sha256_all_16"]) != expected_ids:
        raise ValueError("accepted report does not bind the exact training cohort")
    current_sources = source_hashes()
    for name, expected in ensemble["provenance"]["code_sha256"].items():
        if name in SOURCE_PATHS and current_sources[name] != expected:
            raise ValueError("accepted historical source hash differs: {}".format(name))
    return hgb, ensemble


def verify_training_inputs(cache_dir: str | Path, reference: dict) -> dict[str, str]:
    """Hash only the declared external matches; never discover a new training cohort."""
    expected = reference["provenance"]["input_cache_sha256_all_16"]
    actual = {}
    for match_id in ALLOWED_TRAINING_IDS:
        path = Path(cache_dir) / "{}.npz".format(match_id)
        digest = sha256(path)
        if digest != expected[str(match_id)]:
            raise ValueError("training cache hash differs for {}".format(match_id))
        actual[str(match_id)] = digest
    return actual


def _write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def fit_final_ensemble(cache_dir=DEFAULT_CACHE, model_path=DEFAULT_MODEL,
                       device="cuda", threads: int = 4) -> dict:
    """Refit the externally selected family once on all sixteen allowed matches."""
    path = Path(model_path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError("final artifact/manifest already exists; refusing overwrite")
    if device != "cuda":
        raise ValueError("this frozen final fit requires CUDA; CPU is not an implicit fallback")
    if threads != 4:
        raise ValueError("this frozen final fit uses four CPU threads")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        raise ValueError("frozen CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    before = source_hashes()
    if before["solution/train_tcn_external.py"] != TRAINER_SHA256:
        raise ValueError("accepted TCN trainer bytes have changed")
    hgb_reference, reference = validate_reference_reports()
    input_hashes = verify_training_inputs(cache_dir, reference)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen CUDA device is unavailable")
    torch.set_num_threads(threads)
    start = time.monotonic()
    print("loading exactly 16 externally validated training caches", flush=True)
    caches = load_allowed_matches(str(cache_dir), ALLOWED_TRAINING_IDS,
                                  input_cache_contract("contact"))
    print("fitting HGB-v5 on all 16 matches", flush=True)
    with threadpool_limits(limits=threads):
        values, labels, weights, names = _training_arrays(caches, stride=5)
        if list(names) != hgb_reference["feature_names"]:
            raise ValueError("final HGB feature order differs from accepted report")
        hgb_models = fit_models(values, labels, names, hgb_config(), sample_weights=weights)
        del values, labels, weights
        gc.collect()
        prepared = tuple(prepare_cached_match(cache, SAMPLE_FPS) for cache in caches)
        del caches
        if any(match.feature_names != names for match in prepared):
            raise ValueError("TCN and HGB final feature orders differ")
        normalizer = fit_fold_normalizer(prepared)
        models, histories = [], []
        for seed in TRAINING_SEEDS:
            print("fitting eight-epoch high-capacity TCN, seed {}".format(seed), flush=True)
            model, history = train_sequence_model(prepared, normalizer, tcn_config(seed), device)
            models.append(model.cpu().eval())
            histories.append({"seed": seed, "epochs": history})
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print("finished TCN seed {} after {:.1f} seconds".format(
                seed, time.monotonic() - start), flush=True)
    if source_hashes() != before:
        raise RuntimeError("source changed during training; refusing final artifact publication")
    if verify_training_inputs(cache_dir, reference) != input_hashes:
        raise RuntimeError("training inputs changed during fit")
    manifest = build_ensemble_manifest(input_hashes, extra={
        "source_sha256": before,
        "reference_report_sha256": {"hgb": HGB_REFERENCE_SHA256,
                                     "ensemble": ENSEMBLE_REFERENCE_SHA256},
        "hgb_config": asdict(hgb_config()), "hgb_sample_stride": 5,
        "training_config": [asdict(tcn_config(seed)) for seed in TRAINING_SEEDS],
        "training_history": histories,
        "device": {"type": device, "name": torch.cuda.get_device_name(0),
                   "cpu_threads": threads, "cublas_workspace_config":
                   os.environ["CUBLAS_WORKSPACE_CONFIG"]},
        "dependencies": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "sklearn":
                         importlib.metadata.version("scikit-learn")},
        "elapsed_seconds": time.monotonic() - start,
        "target_labels_used": False,
        "public_success_requirement": {"bootstrap_n": 1000,
                                       "spell_bootstrap_low_strictly_greater_than": 0.75},
        "resampling_contract": "adjacent-native-linear-per-run-10hz-v1",
    })
    artifact = FrozenEnsembleArtifact(hgb_models, tuple(models), normalizer, manifest)
    save_ensemble_artifact(path, artifact)
    digest = ensemble_artifact_sha256(path)
    _write_new_json(manifest_path, {"artifact_sha256": digest, "manifest": manifest})
    print("saved {} sha256 {}".format(path, digest), flush=True)
    return {"path": str(path), "sha256": digest, "manifest_path": str(manifest_path)}


def blend_probability_pairs(pairs) -> tuple[np.ndarray, np.ndarray]:
    if len(pairs) != len(BLEND_WEIGHTS):
        raise ValueError("exactly three probability sources are required")
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("each source requires live and home probability heads")
    reference_shape = np.shape(pairs[0][0])
    if len(reference_shape) != 1:
        raise ValueError("probability arrays must be vectors")
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each source requires live and home probability heads")
        for values in pair:
            values = np.asarray(values)
            if values.shape != reference_shape or not np.isfinite(values).all() \
                    or np.any((values < 0) | (values > 1)):
                raise ValueError("probability sources differ in shape or contain invalid values")
    return tuple(sum(weight * np.asarray(pair[head], dtype=np.float64)
                     for weight, pair in zip(BLEND_WEIGHTS, pairs)) for head in range(2))


def infer_tracking_sequence(sequence, artifact: FrozenEnsembleArtifact) -> np.ndarray:
    """Compute native-frame labels without any target-label dependency."""
    validate_ensemble_artifact(artifact)
    sampled = resample_tracking(sequence, target_fps=SAMPLE_FPS)
    if not sampled.sampled_runs:
        return np.full(sampled.source_n_frames, "X", dtype="<U1")
    batch = prepare_target_features(sampled.sequence, artifact.feature_names)
    n_frames = len(batch.values)
    dummy = PreparedMatch(-1, batch.values, np.full(n_frames, "X", dtype="<U1"),
                          batch.feature_names, batch.valid_runs)
    hgb_pair = predict_probabilities(artifact.hgb_models, batch.values)
    pairs = [hgb_pair]
    for model in artifact.tcn_models:
        pairs.append(predict_prepared_match(model, dummy, artifact.normalizer, "cpu"))
    sampled_alive, sampled_home = blend_probability_pairs(pairs)
    alive = project_probabilities(sampled_alive, sampled)
    home = project_probabilities(sampled_home, sampled)
    codes = decode_runs(compose_emissions(alive, home), valid_runs(sequence.period),
                        sequence.fps, CANONICAL_DURATION_CONFIG)
    return LETTERS[codes]


def _write_new_submission(path: Path, labels: np.ndarray) -> str:
    """Publish validated CSV bytes without replacing an existing output."""
    if path.exists():
        raise FileExistsError("submission already exists: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".ensemble-", suffix=".csv", dir=path.parent)
    os.close(handle)
    try:
        write_validated_submission(temporary, labels)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha256(path)


def run_target_inference(model_path=DEFAULT_MODEL, output_path=DEFAULT_OUTPUT,
                         data_dir=None, expected_sha256=None) -> dict:
    output = Path(output_path)
    record_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or record_path.exists():
        raise FileExistsError("submission/record already exists; refusing overwrite")
    model_digest = ensemble_artifact_sha256(model_path)
    if expected_sha256 is not None and model_digest != expected_sha256:
        raise ValueError("ensemble artifact SHA-256 mismatch")
    artifact = load_ensemble_artifact(model_path, expected_sha256=model_digest)
    frozen_sources = source_hashes()
    if artifact.manifest.get("source_sha256") != frozen_sources:
        raise ValueError("deployment source differs from the artifact's frozen source")
    torch.set_num_threads(4)
    frozen_inputs = target_input_hashes(data_dir)
    sequence = load_bar_ath_sequence(data_dir)
    with threadpool_limits(limits=4):
        labels = infer_tracking_sequence(sequence, artifact)
    # Repeat prediction before any scoring to check deterministic emitted labels.
    with threadpool_limits(limits=4):
        repeated = infer_tracking_sequence(sequence, artifact)
    if not np.array_equal(labels, repeated):
        raise RuntimeError("repeated label-free inference did not reproduce exact labels")
    if source_hashes() != frozen_sources:
        raise RuntimeError("source changed during inference; refusing publication")
    if target_input_hashes(data_dir) != frozen_inputs:
        raise RuntimeError("target inputs changed during inference; refusing publication")
    if ensemble_artifact_sha256(model_path) != model_digest:
        raise RuntimeError("artifact changed during inference; refusing publication")
    digest = _write_new_submission(output, labels)
    record = {"submission": str(output), "sha256": digest,
              "artifact_sha256": model_digest, "target_input_sha256": frozen_inputs,
              "source_sha256": frozen_sources, "frames": len(labels),
              "counts": {label: int(np.sum(labels == label)) for label in LETTERS.tolist()},
              "deterministic_label_reproduction": True, "target_labels_used": False}
    _write_new_json(record_path, record)
    print(json.dumps(record, indent=2), flush=True)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="create the fixed final model from external caches")
    fit.add_argument("--cache", default=str(DEFAULT_CACHE))
    fit.add_argument("--model", default=str(DEFAULT_MODEL))
    infer = commands.add_parser("infer", help="create one frozen target submission")
    infer.add_argument("--model", default=str(DEFAULT_MODEL))
    infer.add_argument("--out", default=str(DEFAULT_OUTPUT))
    infer.add_argument("--data-dir", default=None)
    infer.add_argument("--expected-sha256", default=None)
    args = parser.parse_args(argv)
    if args.command == "fit":
        fit_final_ensemble(args.cache, args.model)
    else:
        run_target_inference(args.model, args.out, args.data_dir, args.expected_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
