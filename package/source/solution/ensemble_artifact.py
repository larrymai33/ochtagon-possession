"""Versioned, create-only persistence for the accepted tracking ensemble.

Artifacts are trusted local pickle/joblib objects, not an untrusted upload format.
The optional expected digest is checked against one byte snapshot before loading it.
"""

from __future__ import annotations

# Pinned Windows wheels require Torch before sklearn-bearing imports.
import torch

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import os
from pathlib import Path
import re
import tempfile

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted

from solution.contact_cache import COMBINED_FEATURE_SCHEMA_VERSION
from solution.external_labels import SKILLCORNER_SOURCE_COMMIT
from solution.temporal import DurationConfig
from solution.train_external import FittedModels, ModelConfig
from solution.train_tcn_external import (
    ALLOWED_DEVELOPMENT_MATCH_IDS,
    FeatureNormalizer,
    TinyTemporalConvNet,
    _DilatedResidualBlock,
)


ARTIFACT_VERSION = "tracking-hgb-two-seed-tcn-ensemble-v1"
MANIFEST_SCHEMA_VERSION = "tracking-hgb-two-seed-tcn-manifest-v1"
ALLOWED_TRAINING_IDS = ALLOWED_DEVELOPMENT_MATCH_IDS
TRAINING_SEEDS = (20261023, 20261123)
SAMPLE_FPS = 10.0
BLEND_WEIGHTS = (0.5, 0.25, 0.25)
CANONICAL_DURATION_CONFIG = DurationConfig((0.1, 0.1, 2.5), 1.4, 2.0)
FEATURE_COUNT = 253
TCN_DILATIONS = (1, 2, 4, 8, 16, 32, 64)
HGB_CONFIG = ModelConfig(max_leaf_nodes=63, min_samples_leaf=30)


@dataclass(frozen=True)
class FrozenEnsembleArtifact:
    hgb_models: FittedModels
    tcn_models: tuple[TinyTemporalConvNet, TinyTemporalConvNet]
    normalizer: FeatureNormalizer
    manifest: dict
    artifact_version: str = ARTIFACT_VERSION

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.normalizer.feature_names

    @property
    def duration_config(self) -> DurationConfig:
        return CANONICAL_DURATION_CONFIG

    @property
    def blend_weights(self) -> tuple[float, float, float]:
        return BLEND_WEIGHTS

    @property
    def sample_fps(self) -> float:
        return SAMPLE_FPS


def _fixed_manifest() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "training_ids": list(ALLOWED_TRAINING_IDS),
        "seeds": list(TRAINING_SEEDS),
        "source_commit": SKILLCORNER_SOURCE_COMMIT,
        "input_feature_schema": COMBINED_FEATURE_SCHEMA_VERSION,
        "blend": list(BLEND_WEIGHTS),
        "decoder": {
            "min_duration_s": list(CANONICAL_DURATION_CONFIG.min_duration_s),
            "team_switch_penalty": CANONICAL_DURATION_CONFIG.team_switch_penalty,
            "dead_switch_penalty": CANONICAL_DURATION_CONFIG.dead_switch_penalty,
        },
        "sample_fps": SAMPLE_FPS,
    }


def _validate_hashes(input_sha256: dict) -> None:
    if not isinstance(input_sha256, dict) or set(input_sha256) != {
        str(value) for value in ALLOWED_TRAINING_IDS
    }:
        raise ValueError("input_sha256 must cover exactly the 16 training IDs")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in input_sha256.values()):
        raise ValueError("input_sha256 contains an invalid SHA-256 digest")


def build_ensemble_manifest(input_sha256: dict[str, str], *, extra: dict | None = None) -> dict:
    """Build canonical provenance; extra metadata cannot replace required fields.

    ``input_sha256`` maps each decimal match-ID string to its cache-byte digest.
    Required fields use JSON-compatible lists, strings, and dictionaries. Additional
    provenance is retained without interpreting it as an inference configuration.
    """
    _validate_hashes(input_sha256)
    manifest = _fixed_manifest()
    manifest["input_sha256"] = dict(sorted(input_sha256.items()))
    if extra is not None:
        if not isinstance(extra, dict) or set(extra) & set(manifest):
            raise ValueError("extra manifest metadata cannot replace canonical fields")
        manifest.update(copy.deepcopy(extra))
    return manifest


def _validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("ensemble manifest must be a dict")
    for key, expected in _fixed_manifest().items():
        if manifest.get(key) != expected:
            raise ValueError("ensemble manifest field {!r} differs".format(key))
    _validate_hashes(manifest.get("input_sha256"))


def _validate_convolution(layer, inputs: int, outputs: int, kernel: int,
                          dilation: int = 1) -> None:
    padding = dilation if kernel == 3 else 0
    if type(layer) is not torch.nn.Conv1d or (
        layer.in_channels != inputs or layer.out_channels != outputs
        or layer.kernel_size != (kernel,) or layer.stride != (1,)
        or layer.padding != (padding,) or layer.dilation != (dilation,)
        or layer.groups != 1 or layer.padding_mode != "zeros" or layer.bias is None
        or tuple(layer.weight.shape) != (outputs, inputs, kernel)
        or tuple(layer.bias.shape) != (outputs,)
    ):
        raise ValueError("TCN convolution differs from the accepted architecture")


def _validate_tcn(model: TinyTemporalConvNet) -> None:
    if type(model) is not TinyTemporalConvNet or set(model._modules) != {
        "input_projection", "blocks", "live_head", "team_head"
    }:
        raise ValueError("TCN must be the accepted TinyTemporalConvNet")
    _validate_convolution(model.input_projection, FEATURE_COUNT * 2, 96, 1)
    _validate_convolution(model.live_head, 96, 1, 1)
    _validate_convolution(model.team_head, 96, 1, 1)
    if type(model.blocks) is not torch.nn.Sequential or len(model.blocks) != 7:
        raise ValueError("TCN must contain exactly seven residual blocks")
    for block, dilation in zip(model.blocks, TCN_DILATIONS):
        if type(block) is not _DilatedResidualBlock or set(block._modules) != {
            "conv", "norm", "activation", "dropout"
        }:
            raise ValueError("TCN residual block structure differs")
        _validate_convolution(block.conv, 96, 96, 3, dilation)
        norm = block.norm
        if type(norm) is not torch.nn.BatchNorm1d or (
            norm.num_features != 96 or norm.eps != 1e-5 or norm.momentum != 0.1
            or not norm.affine or not norm.track_running_stats
            or norm.running_mean is None or norm.running_var is None
            or tuple(norm.running_mean.shape) != (96,) or tuple(norm.running_var.shape) != (96,)
            or bool(torch.any(norm.running_var < 0))
        ):
            raise ValueError("TCN normalization differs from accepted BatchNorm")
        if type(block.activation) is not torch.nn.GELU or block.activation.approximate != "none":
            raise ValueError("TCN activation differs")
        if type(block.dropout) is not torch.nn.Dropout or (
            block.dropout.p != 0.1 or block.dropout.inplace
        ):
            raise ValueError("TCN dropout differs")
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("TCN parameters and buffers must be finite")


def validate_ensemble_artifact(artifact: FrozenEnsembleArtifact) -> None:
    """Reject artifacts outside the fixed external-development family."""
    if type(artifact) is not FrozenEnsembleArtifact or artifact.artifact_version != ARTIFACT_VERSION:
        raise ValueError("unsupported frozen ensemble artifact/version")
    _validate_manifest(artifact.manifest)
    if type(artifact.normalizer) is not FeatureNormalizer:
        raise ValueError("ensemble requires a FeatureNormalizer")
    names = artifact.normalizer.feature_names
    if not isinstance(names, tuple) or len(names) != FEATURE_COUNT or any(
        not isinstance(name, str) or not name for name in names
    ) or len(set(names)) != FEATURE_COUNT:
        raise ValueError("ensemble requires 253 unique ordered feature names")
    for field in ("mean", "scale"):
        value = np.asarray(getattr(artifact.normalizer, field))
        if value.shape != (FEATURE_COUNT,) or not np.isfinite(value).all():
            raise ValueError("normalizer {} must be a finite 253-vector".format(field))
        if field == "scale" and np.any(value <= 0):
            raise ValueError("normalizer scale must be strictly positive")
    hgb = artifact.hgb_models
    if type(hgb) is not FittedModels or hgb.feature_names != names:
        raise ValueError("HGB and normalizer feature schema differ")
    if type(hgb.config) is not ModelConfig or hgb.config != HGB_CONFIG:
        raise ValueError("HGB configuration differs from accepted v5")
    expected_parameters = {**asdict(HGB_CONFIG), "early_stopping": False}
    for model in (hgb.live_model, hgb.team_model):
        if type(model) is not HistGradientBoostingClassifier:
            raise ValueError("HGB components must be HistGradientBoostingClassifier")
        try:
            check_is_fitted(model)
        except (TypeError, ValueError) as error:
            raise ValueError("HGB component is not fitted") from error
        if getattr(model, "n_features_in_", None) != FEATURE_COUNT or not np.array_equal(
            getattr(model, "classes_", None), np.array([0, 1])
        ):
            raise ValueError("fitted HGB feature width/classes differ")
        parameters = model.get_params(deep=False)
        if any(parameters.get(key) != value for key, value in expected_parameters.items()):
            raise ValueError("fitted HGB parameters differ from accepted v5")
    if not isinstance(artifact.tcn_models, tuple) or len(artifact.tcn_models) != 2:
        raise ValueError("ensemble requires exactly two TCN models")
    for model in artifact.tcn_models:
        _validate_tcn(model)


def save_ensemble_artifact(path: str | os.PathLike, artifact: FrozenEnsembleArtifact) -> str:
    """Persist CPU/eval copies atomically, failing if the destination already exists.

    The temporary file and destination share a directory. Publishing a hard link
    is an atomic exclusive operation on supported filesystems; an unsupported
    filesystem fails without a destination instead of falling back to overwrite.
    """
    validate_ensemble_artifact(artifact)
    destination = Path(path)
    if os.path.lexists(destination):
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    persistent = copy.deepcopy(artifact)
    persistent = replace(persistent, tcn_models=tuple(
        model.cpu().eval() for model in persistent.tcn_models
    ))
    validate_ensemble_artifact(persistent)
    descriptor, temporary = tempfile.mkstemp(prefix=".ensemble-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            joblib.dump(persistent, handle, compress=3)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return str(destination)


def load_ensemble_artifact(path: str | os.PathLike, expected_sha256: str | None = None
                           ) -> FrozenEnsembleArtifact:
    """Verify one immutable byte snapshot before deserializing trusted local data."""
    snapshot = Path(path).read_bytes()
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
        if hashlib.sha256(snapshot).hexdigest() != expected_sha256:
            raise ValueError("ensemble artifact SHA-256 mismatch")
    artifact = joblib.load(io.BytesIO(snapshot))
    validate_ensemble_artifact(artifact)
    if any(value.device.type != "cpu" for model in artifact.tcn_models
           for value in model.state_dict().values()):
        raise ValueError("persisted TCN parameters must be on CPU")
    if any(model.training for model in artifact.tcn_models):
        raise ValueError("persisted TCN models must use evaluation mode")
    return artifact


def ensemble_artifact_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
