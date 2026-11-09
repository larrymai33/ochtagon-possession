"""Label-free inference for the externally trained possession model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm
from solution.contact_cache import COMBINED_FEATURE_SCHEMA_VERSION
from solution.contact_features import extract_contact_features
from solution.model_artifact import load_artifact
from solution.target_adapter import load_bar_ath_sequence
from solution.tracking_features import (
    FeatureBatch,
    TrackingSequence,
    add_temporal_context,
    extract_features,
)
from solution.temporal import compose_emissions, decode_runs
from solution.train_external import predict_probabilities, source_visibility_view
from starter.baseline import write_submission


CODE_LABELS = np.array(("H", "A", "D", "X"), dtype="<U1")
DEFAULT_ARTIFACT = os.path.join(ROOT, "data", "cache", "external_model", "frozen.joblib")
DEFAULT_OUTPUT = os.path.join(ROOT, "evaluation", "results", "external_tracking.csv")


@dataclass(frozen=True)
class InferenceResult:
    output_path: str
    sha256: str
    counts: tuple[tuple[str, int], ...]


def _contextual_target_batch(values: np.ndarray, names: tuple[str, ...],
                             runs: tuple[tuple[int, int], ...], fps: float,
                             schema_version: str) -> FeatureBatch:
    visible = source_visibility_view(values, names)
    contextual, contextual_names = add_temporal_context(visible, names, runs, fps)
    return FeatureBatch(
        values=contextual,
        feature_names=contextual_names,
        valid_runs=runs,
        schema_version=schema_version,
    )


def prepare_target_features(
        sequence: TrackingSequence,
        expected_feature_names: tuple[str, ...] | None = None) -> FeatureBatch:
    """Reproduce the exact base-only or base+contact external-training transform."""
    base = extract_features(sequence)
    base_prepared = _contextual_target_batch(
        base.values, base.feature_names, base.valid_runs, sequence.fps,
        base.schema_version,
    )
    if expected_feature_names is None \
            or tuple(expected_feature_names) == base_prepared.feature_names:
        return base_prepared

    contact = extract_contact_features(sequence)
    if contact.valid_runs != base.valid_runs:
        raise ValueError("base/contact target valid runs differ")
    combined_names = base.feature_names + contact.feature_names
    combined_values = np.concatenate((base.values, contact.values), axis=1)
    combined_prepared = _contextual_target_batch(
        combined_values, combined_names, base.valid_runs, sequence.fps,
        COMBINED_FEATURE_SCHEMA_VERSION,
    )
    if tuple(expected_feature_names) != combined_prepared.feature_names:
        raise ValueError("feature schema mismatch between artifact and target inference")
    return combined_prepared


def infer_sequence(sequence: TrackingSequence, artifact) -> np.ndarray:
    """Predict canonical H/A/D/X labels without reading any target labels."""
    prepared = prepare_target_features(
        sequence, expected_feature_names=tuple(artifact.feature_names)
    )
    expected_input_schema = artifact.manifest.get("input_feature_schema")
    if expected_input_schema is not None \
            and expected_input_schema != prepared.schema_version:
        raise ValueError("artifact input feature schema {} != reproduced {}".format(
            expected_input_schema, prepared.schema_version
        ))
    if tuple(artifact.feature_names) != prepared.feature_names:
        raise ValueError("feature schema mismatch between artifact and target inference")
    rows = np.concatenate([
        np.arange(start, stop, dtype=np.int64)
        for start, stop in prepared.valid_runs
    ]) if prepared.valid_runs else np.empty(0, dtype=np.int64)
    p_alive = np.full(len(prepared.values), 0.5, dtype=np.float64)
    p_home = np.full(len(prepared.values), 0.5, dtype=np.float64)
    if len(rows):
        p_alive[rows], p_home[rows] = predict_probabilities(
            artifact,
            prepared.values[rows],
        )
    emissions = compose_emissions(
        p_alive,
        p_home,
        temperature=artifact.temperature,
    )
    codes = decode_runs(
        emissions,
        prepared.valid_runs,
        sequence.fps,
        artifact.duration_config,
    )
    return CODE_LABELS[codes]


def write_validated_submission(path: str, labels: np.ndarray) -> str:
    """Write the canonical CSV and reject any structural contract violation."""
    output = write_submission(path, labels)
    table = pm.read_label_csv(output)
    reason = pm.validate_submission(table, len(labels))
    if reason is not None:
        raise ValueError("submission validation failed: {}".format(reason))
    return output


def submission_sha256(path: str) -> str:
    """Hash the exact submission bytes for reproducibility checks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_target_inference(model_path: str, output_path: str,
                         data_dir: str | None = None) -> InferenceResult:
    """Load target tracking and a frozen external artifact, then write one submission."""
    sequence = load_bar_ath_sequence(data_dir)
    artifact = load_artifact(model_path)
    labels = infer_sequence(sequence, artifact)
    path = write_validated_submission(output_path, labels)
    counts = tuple(
        (label, int(np.sum(labels == label)))
        for label in CODE_LABELS.tolist()
    )
    return InferenceResult(path, submission_sha256(path), counts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_ARTIFACT,
                        help="frozen externally trained model artifact")
    parser.add_argument("--out", default=DEFAULT_OUTPUT,
                        help="canonical frame,label CSV to write")
    parser.add_argument("--data-dir", default=None,
                        help="optional root of the gated target tracking bundle")
    args = parser.parse_args(argv)

    result = run_target_inference(args.model, args.out, args.data_dir)
    print("wrote {}".format(result.output_path))
    print("sha256 {}".format(result.sha256))
    print("counts {}".format(dict(result.counts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
