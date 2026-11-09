"""Versioned persistence for the externally trained two-stage model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os

import joblib

from solution.temporal import DurationConfig


ARTIFACT_VERSION = "external-possession-v1"


@dataclass(frozen=True)
class ExternalArtifact:
    live_model: object
    team_model: object
    feature_names: tuple[str, ...]
    duration_config: DurationConfig
    temperature: float
    manifest: dict
    artifact_version: str = ARTIFACT_VERSION


def save_artifact(path: str, artifact: ExternalArtifact) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    return path


def load_artifact(path: str, expected_feature_names: tuple[str, ...] | None = None
                  ) -> ExternalArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, ExternalArtifact):
        raise ValueError("{} is not an ExternalArtifact".format(path))
    if artifact.artifact_version != ARTIFACT_VERSION:
        raise ValueError("artifact version {} != expected {}".format(
            artifact.artifact_version, ARTIFACT_VERSION))
    if expected_feature_names is not None \
            and tuple(artifact.feature_names) != tuple(expected_feature_names):
        raise ValueError("feature schema mismatch between artifact and inference")
    return artifact


def artifact_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
