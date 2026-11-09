"""External-only temporal convolution experiment for SkillCorner possession labels.

This module is intentionally separate from the frozen HGB training/inference path.  It may read
only the explicitly allowed SkillCorner development caches; it never reads BAR-ATH labels or
reports.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution.external_labels import (
    FINAL_TEST_MATCH_IDS,
    SKILLCORNER_SOURCE_COMMIT,
    development_folds,
)
from solution.contact_cache import (
    BASE_FEATURE_WIDTH,
    COMBINED_CACHE_ADAPTER_VERSION,
    COMBINED_FEATURE_SCHEMA_VERSION,
    COMBINED_FEATURE_WIDTH,
    CONTACT_FEATURE_SCHEMA_VERSION,
    SOURCE_ADAPTER_VERSION,
    load_combined_feature_cache,
)
from solution.skillcorner_adapter import (
    ADAPTER_VERSION,
    SKILLCORNER_FPS,
    load_feature_cache,
)
from solution.train_external import (
    DEFAULT_CACHE,
    DEFAULT_WORK,
    _cache_paths,
    evaluate_decoder,
    source_visibility_view,
    validate_development_ids,
)
from solution.tracking_features import FEATURE_SCHEMA_VERSION, add_temporal_context


ALLOWED_DEVELOPMENT_MATCH_IDS = (
    1874553, 1886347, 1899585, 1925299,
    1953632, 1959846, 1986691, 1996435,
    2006229, 2006363, 2007448, 2007721,
    2011166, 2013725, 2015213, 2016236,
)
CONTEXT_OFFSETS_S = (0.3, 1.0, 3.0)
BASE_CONTEXT_FEATURE_COUNT = 181
CONTACT_CONTEXT_FEATURE_COUNT = 253


@dataclass(frozen=True)
class InputCacheContract:
    """One explicitly selected cache schema and its immutable audit expectations."""

    name: str
    schema_version: str
    raw_feature_count: int
    context_feature_count: int
    manifest_expectations: dict[str, str]
    baseline_report_name: str
    experiment_name: str


def input_cache_contract(name: str) -> InputCacheContract:
    """Resolve only the two audited cache contracts; never infer a schema from files."""
    if name == "base":
        return InputCacheContract(
            name="base",
            schema_version=FEATURE_SCHEMA_VERSION,
            raw_feature_count=BASE_FEATURE_WIDTH,
            context_feature_count=BASE_CONTEXT_FEATURE_COUNT,
            manifest_expectations={
                "source_commit": SKILLCORNER_SOURCE_COMMIT,
                "schema_version": FEATURE_SCHEMA_VERSION,
                "adapter_version": ADAPTER_VERSION,
            },
            baseline_report_name="hgb_tracking_context_v2.json",
            experiment_name="tcn_tracking_context_v1",
        )
    if name == "contact":
        return InputCacheContract(
            name="contact",
            schema_version=COMBINED_FEATURE_SCHEMA_VERSION,
            raw_feature_count=COMBINED_FEATURE_WIDTH,
            context_feature_count=CONTACT_CONTEXT_FEATURE_COUNT,
            manifest_expectations={
                "source_commit": SKILLCORNER_SOURCE_COMMIT,
                "schema_version": COMBINED_FEATURE_SCHEMA_VERSION,
                "base_schema_version": FEATURE_SCHEMA_VERSION,
                "contact_schema_version": CONTACT_FEATURE_SCHEMA_VERSION,
                "source_adapter_version": SOURCE_ADAPTER_VERSION,
                "cache_adapter_version": COMBINED_CACHE_ADAPTER_VERSION,
            },
            baseline_report_name="hgb_contact_context_segment_balanced_capacity_v5.json",
            experiment_name="tcn_contact_context_v1",
        )
    raise ValueError("input schema must be explicitly 'base' or 'contact', got {!r}".format(
        name
    ))


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedMatch:
    match_id: int
    values: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    valid_runs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SequenceConfig:
    hidden_channels: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    dropout: float = 0.1

    def __post_init__(self):
        if self.hidden_channels < 1 or not self.dilations or min(self.dilations) < 1:
            raise ValueError("hidden_channels and dilations must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20261023
    epochs: int = 5
    batch_size: int = 16
    window_frames: int = 1024
    stride_frames: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    transition_radius_frames: int = 10
    transition_boost: float = 4.0
    segment_weighting: str = "none"
    segment_weight_cap: float = 8.0
    sequence: SequenceConfig = SequenceConfig(hidden_channels=48)

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.window_frames < 1 \
                or self.stride_frames < 1 or self.learning_rate <= 0 \
                or self.weight_decay < 0 or self.transition_radius_frames < 0 \
                or self.transition_boost < 1 or self.segment_weight_cap < 1 \
                or self.segment_weighting not in ("none", "sqrt_inverse"):
            raise ValueError("invalid temporal experiment configuration")


def _paired_feature_indices(feature_names: tuple[str, ...]):
    positions = {name: index for index, name in enumerate(feature_names)}
    for index, name in enumerate(feature_names):
        if name.startswith("home_"):
            peer = "away_" + name[len("home_"):]
            if peer in positions:
                yield index, positions[peer]


def fit_normalizer(arrays, feature_names: tuple[str, ...]) -> FeatureNormalizer:
    """Fit finite-value moments, pooling explicit home/away pairs for swap symmetry."""
    names = tuple(feature_names)
    total = np.zeros(len(names), dtype=np.float64)
    squared = np.zeros(len(names), dtype=np.float64)
    count = np.zeros(len(names), dtype=np.int64)
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("normalizer arrays do not match feature schema")
        finite = np.isfinite(values)
        total += np.where(finite, values, 0.0).sum(axis=0)
        squared += np.where(finite, values * values, 0.0).sum(axis=0)
        count += finite.sum(axis=0)
    if np.any(count == 0):
        missing = [names[i] for i in np.flatnonzero(count == 0)]
        raise ValueError("features have no finite training values: {}".format(missing))
    for home, away in _paired_feature_indices(names):
        pooled_count = count[home] + count[away]
        pooled_total = total[home] + total[away]
        pooled_squared = squared[home] + squared[away]
        count[home] = count[away] = pooled_count
        total[home] = total[away] = pooled_total
        squared[home] = squared[away] = pooled_squared
    mean = total / count
    variance = np.maximum(0.0, squared / count - mean * mean)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return FeatureNormalizer(mean.astype(np.float32), scale.astype(np.float32), names)


def normalize_features(values: np.ndarray, normalizer: FeatureNormalizer) -> np.ndarray:
    """Standardize finite values and append one observed-value mask per feature."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != len(normalizer.feature_names):
        raise ValueError("values do not match normalizer feature schema")
    finite = np.isfinite(source)
    standardized = (source - normalizer.mean) / normalizer.scale
    standardized = np.clip(standardized, -8.0, 8.0)
    standardized[~finite] = 0.0
    return np.concatenate((standardized, finite.astype(np.float32)), axis=1)


def prepare_cached_match(cached, fps: float = 10.0,
                         context_offsets_s: tuple[float, ...] = CONTEXT_OFFSETS_S
                         ) -> PreparedMatch:
    """Apply the target-compatible visibility contract, then add run-safe context."""
    names = tuple(cached.feature_names)
    visible = source_visibility_view(cached.values, names)
    contextual, context_names = add_temporal_context(
        visible, names, tuple(cached.valid_runs), fps, context_offsets_s
    )
    return PreparedMatch(
        match_id=int(cached.match_id),
        values=contextual,
        labels=np.asarray(cached.labels).astype("<U1", copy=True),
        feature_names=context_names,
        valid_runs=tuple((int(start), int(stop)) for start, stop in cached.valid_runs),
    )


def validate_exact_development_ids(match_ids) -> tuple[int, ...]:
    """Require the declared 16-match experiment cohort, not merely any 16 caches."""
    actual = tuple(sorted(int(match_id) for match_id in match_ids))
    if actual != ALLOWED_DEVELOPMENT_MATCH_IDS:
        missing = sorted(set(ALLOWED_DEVELOPMENT_MATCH_IDS) - set(actual))
        unexpected = sorted(set(actual) - set(ALLOWED_DEVELOPMENT_MATCH_IDS))
        raise ValueError("development caches differ from exact allowlist; missing={} unexpected={}".format(
            missing, unexpected))
    return actual


def validate_loaded_match_ids(requested_ids, caches) -> None:
    """Reject a cache whose internal match ID does not match its selected filename."""
    requested = tuple(int(match_id) for match_id in requested_ids)
    loaded = tuple(int(cached.match_id) for cached in caches)
    if len(requested) != len(loaded) or requested != loaded:
        raise ValueError("cache internal match_id sequence {} != requested {}".format(
            list(loaded), list(requested)))


def validate_cache_manifest(path: str, contract: InputCacheContract) -> None:
    """Require every contract field before a schema-specific loader touches the cache."""
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(contract.manifest_expectations) - set(archive.files))
        if missing:
            raise ValueError("cache manifest is missing fields {} for {}: {}".format(
                missing, contract.name, path
            ))
        for field, expected in sorted(contract.manifest_expectations.items()):
            actual = str(archive[field].item())
            if actual != expected:
                raise ValueError(
                    "cache manifest {} mismatch for {}: {!r} != {!r}: {}".format(
                        field, contract.name, actual, expected, path
                    )
                )


def load_allowed_matches(cache_dir: str, requested_ids,
                         contract: InputCacheContract) -> tuple:
    """Load the exact allowlist through only the explicitly requested schema loader."""
    requested = validate_exact_development_ids(requested_ids)
    validate_development_ids(requested)
    paths = _cache_paths(cache_dir)
    validate_exact_development_ids(paths)
    loader = (load_feature_cache if contract.name == "base"
              else load_combined_feature_cache)
    loaded = []
    for match_id in requested:
        path = paths[match_id]
        validate_cache_manifest(path, contract)
        cached = loader(path)
        if int(cached.match_id) != match_id:
            raise ValueError("cache internal match_id {} != requested {}: {}".format(
                cached.match_id, match_id, path
            ))
        if cached.schema_version != contract.schema_version:
            raise ValueError("loaded cache schema {!r} != requested {!r}: {}".format(
                cached.schema_version, contract.schema_version, path
            ))
        if cached.values.ndim != 2 or cached.values.shape[1] != contract.raw_feature_count:
            raise ValueError("{} cache feature width {} != expected {}: {}".format(
                contract.name,
                cached.values.shape[1] if cached.values.ndim == 2 else cached.values.shape,
                contract.raw_feature_count,
                path,
            ))
        if len(cached.feature_names) != contract.raw_feature_count:
            raise ValueError("{} cache feature names do not match expected width: {}".format(
                contract.name, path
            ))
        loaded.append(cached)
    validate_loaded_match_ids(requested, loaded)
    return tuple(loaded)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_provenance(cache_paths: dict[int, str], code_paths: dict[str, str],
                         feature_names: tuple[str, ...], context_offsets_s: tuple[float, ...],
                         contract: InputCacheContract,
                         ) -> dict:
    """Hash every external cache and code dependency needed to reproduce the experiment."""
    names = tuple(str(name) for name in feature_names)
    feature_text = json.dumps(names, separators=(",", ":"))
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    return {
        "skillcorner_source_commit": SKILLCORNER_SOURCE_COMMIT,
        "input_cache_contract": asdict(contract),
        "feature_schema_version": contract.schema_version,
        "allowed_development_match_ids": list(ALLOWED_DEVELOPMENT_MATCH_IDS),
        "input_cache_sha256": {
            str(match_id): _file_sha256(path)
            for match_id, path in sorted(cache_paths.items())
        },
        "code_sha256": {
            name: _file_sha256(path) for name, path in sorted(code_paths.items())
        },
        "git_revision": git_revision,
        "feature_count": len(names),
        "feature_names_sha256": hashlib.sha256(feature_text.encode()).hexdigest(),
        "context_offsets_s": [float(value) for value in context_offsets_s],
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }


def fit_fold_normalizer(matches: tuple[PreparedMatch, ...]) -> FeatureNormalizer:
    """Fit preprocessing on training matches and graded runs only."""
    if not matches:
        raise ValueError("a fold needs at least one training match")
    names = matches[0].feature_names
    if any(match.feature_names != names for match in matches):
        raise ValueError("feature schema differs between training matches")
    arrays = (match.values[start:stop]
              for match in matches for start, stop in match.valid_runs)
    return fit_normalizer(arrays, names)


def build_windows(runs: tuple[tuple[int, int], ...], window_frames: int,
                  stride_frames: int) -> tuple[tuple[int, int], ...]:
    """Return fixed-size windows within runs, adding a tail-aligned window when needed."""
    if window_frames < 1 or stride_frames < 1:
        raise ValueError("window and stride must be positive")
    windows = []
    previous_stop = 0
    for start, stop in runs:
        start, stop = int(start), int(stop)
        if start < previous_stop or stop <= start:
            raise ValueError("runs must be ordered, disjoint, and non-empty")
        previous_stop = stop
        if stop - start < window_frames:
            continue
        last = stop - window_frames
        starts = list(range(start, last + 1, stride_frames))
        if starts[-1] != last:
            starts.append(last)
        windows.extend((position, position + window_frames) for position in starts)
    return tuple(windows)


def transition_weights(labels: np.ndarray, runs: tuple[tuple[int, int], ...],
                       radius_frames: int, boost: float) -> np.ndarray:
    """Upweight a symmetric neighborhood of true changes without crossing run gaps."""
    values = np.asarray(labels)
    if values.ndim != 1 or radius_frames < 0 or boost < 1.0:
        raise ValueError("labels, radius_frames, or boost are invalid")
    weights = np.ones(len(values), dtype=np.float32)
    for start, stop in runs:
        start, stop = int(start), int(stop)
        changes = np.flatnonzero(values[start + 1:stop] != values[start:stop - 1]) + start + 1
        for change in changes:
            low = max(start, int(change) - radius_frames)
            high = min(stop, int(change) + radius_frames)
            weights[low:high] = boost
    return weights


def segment_frame_weights(labels: np.ndarray, runs: tuple[tuple[int, int], ...],
                          mode: str, cap: float) -> np.ndarray:
    """Return deterministic duration weights, normalized over valid frames then capped."""
    values = np.asarray(labels).astype("<U1")
    if values.ndim != 1 or mode not in ("none", "sqrt_inverse") or cap < 1:
        raise ValueError("invalid labels, segment weighting mode, or cap")
    weights = np.ones(len(values), dtype=np.float32)
    if mode == "none":
        return weights
    valid = np.zeros(len(values), dtype=bool)
    previous_stop = 0
    for start, stop in runs:
        start, stop = int(start), int(stop)
        if start < previous_stop or start < 0 or stop <= start or stop > len(values):
            raise ValueError("runs must be ordered, disjoint, non-empty, and in bounds")
        previous_stop = stop
        valid[start:stop] = True
        changes = np.flatnonzero(values[start + 1:stop] != values[start:stop - 1]) + start + 1
        bounds = np.concatenate(([start], changes, [stop]))
        for index in range(len(bounds) - 1):
            low, high = int(bounds[index]), int(bounds[index + 1])
            weights[low:high] = 1.0 / np.sqrt(float(high - low))
    if not valid.any():
        raise ValueError("at least one valid run is required")
    weights[valid] /= float(weights[valid].mean())
    weights[valid] = np.minimum(weights[valid], float(cap))
    return weights


def class_balance_weights(label_arrays) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (negative, positive) weights for live and conditional-home BCE heads."""
    labels = np.concatenate([np.asarray(values).astype("<U1") for values in label_arrays])
    if not len(labels) or np.any(~np.isin(labels, ("H", "A", "D"))):
        raise ValueError("class weights require non-empty H/A/D labels")

    def balanced(binary):
        counts = np.bincount(np.asarray(binary, dtype=np.int8), minlength=2)
        if np.any(counts == 0):
            raise ValueError("both classes must be present")
        total = float(counts.sum())
        return total / (2.0 * counts[0]), total / (2.0 * counts[1])

    live = labels != "D"
    team_labels = labels[live]
    return balanced(live), balanced(team_labels == "H")


def training_loss_class_weights(label_arrays, base_weight_arrays=None
                                ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Balance live/dead; team swapping makes unit H/A weights the unbiased contract."""
    labels = np.concatenate([np.asarray(values).astype("<U1") for values in label_arrays])
    if base_weight_arrays is None:
        base = np.ones(len(labels), dtype=np.float64)
    else:
        base = np.concatenate([np.asarray(values, dtype=np.float64)
                               for values in base_weight_arrays])
    if base.shape != (len(labels),) or not np.isfinite(base).all() or np.any(base <= 0):
        raise ValueError("effective frame weights must be a finite positive label-length vector")
    live = labels != "D"
    totals = (float(base[~live].sum()), float(base[live].sum()))
    if min(totals) <= 0:
        raise ValueError("live/dead classes must both have positive effective mass")
    total = sum(totals)
    live_weights = (total / (2.0 * totals[0]), total / (2.0 * totals[1]))
    return live_weights, (1.0, 1.0)


def _label_codes(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels).astype("<U1")
    result = np.full(len(values), -1, dtype=np.int64)
    for code, label in enumerate(("H", "A", "D")):
        result[values == label] = code
    if np.any(result < 0):
        raise ValueError("sequence windows may contain only H/A/D labels")
    return result


class SequenceWindowDataset(Dataset):
    """Fixed windows cut strictly inside published valid runs."""
    def __init__(self, matches: tuple[PreparedMatch, ...], normalizer: FeatureNormalizer,
                 window_frames: int, stride_frames: int, transition_radius_frames: int,
                 transition_boost: float, segment_weighting: str = "none",
                 segment_weight_cap: float = 8.0):
        self.matches = tuple(matches)
        self.normalizer = normalizer
        self.windows = []
        self.weights = []
        for match_index, match in enumerate(self.matches):
            if match.feature_names != normalizer.feature_names:
                raise ValueError("match and normalizer feature schemas differ")
            match_weights = transition_weights(
                match.labels, match.valid_runs, transition_radius_frames, transition_boost
            )
            match_weights *= segment_frame_weights(
                match.labels, match.valid_runs, segment_weighting, segment_weight_cap
            )
            self.weights.append(match_weights)
            self.windows.extend((match_index, start, stop) for start, stop in build_windows(
                match.valid_runs, window_frames, stride_frames
            ))
        if not self.windows:
            raise ValueError("no full training windows fit inside the valid runs")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        match_index, start, stop = self.windows[index]
        match = self.matches[match_index]
        values = normalize_features(match.values[start:stop], self.normalizer).T.copy()
        labels = _label_codes(match.labels[start:stop])
        weights = self.weights[match_index][start:stop]
        return (torch.from_numpy(values), torch.from_numpy(labels),
                torch.from_numpy(np.array(weights, copy=True)))


def team_swap_channel_index(feature_names: tuple[str, ...]) -> tuple[int, ...]:
    """Permutation swapping paired normalized values and their appended finite masks."""
    names = tuple(feature_names)
    positions = {name: index for index, name in enumerate(names)}
    permutation = list(range(2 * len(names)))
    for home, away in _paired_feature_indices(names):
        permutation[home], permutation[away] = away, home
        permutation[home + len(names)], permutation[away + len(names)] = (
            away + len(names), home + len(names)
        )
    return tuple(permutation)


def swap_batch_teams(values: torch.Tensor, labels: torch.Tensor,
                     channel_index: tuple[int, ...], rows: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap selected batch rows and exchange H/A targets, leaving D unchanged."""
    if values.ndim != 3 or labels.ndim != 2 or values.shape[0] != labels.shape[0] \
            or values.shape[2] != labels.shape[1] or len(channel_index) != values.shape[1] \
            or rows.shape != (values.shape[0],):
        raise ValueError("team-swap tensors have incompatible shapes")
    swapped_values = values.clone()
    swapped_labels = labels.clone()
    if torch.any(rows):
        index = torch.as_tensor(channel_index, dtype=torch.long, device=values.device)
        swapped_values[rows] = values[rows][:, index, :]
        original = labels[rows]
        replacement = original.clone()
        replacement[original == 0] = 1
        replacement[original == 1] = 0
        swapped_labels[rows] = replacement
    return swapped_values, swapped_labels


def sequence_loss(live_logits: torch.Tensor, team_logits: torch.Tensor,
                  labels: torch.Tensor, frame_weights: torch.Tensor,
                  live_class_weights: tuple[float, float] = (1.0, 1.0),
                  team_class_weights: tuple[float, float] = (1.0, 1.0)) -> torch.Tensor:
    """Two-head BCE; the conditional team head is scored only on H/A frames."""
    if live_logits.shape != labels.shape or team_logits.shape != labels.shape \
            or frame_weights.shape != labels.shape:
        raise ValueError("loss tensors must have identical [batch, frames] shapes")
    live_target = (labels != 2).to(live_logits.dtype)
    home_target = (labels == 0).to(team_logits.dtype)
    live_balance = torch.where(live_target > 0.5,
                               live_logits.new_tensor(live_class_weights[1]),
                               live_logits.new_tensor(live_class_weights[0]))
    live_weight = frame_weights * live_balance
    live_loss = (F.binary_cross_entropy_with_logits(
        live_logits, live_target, reduction="none") * live_weight).sum() / live_weight.sum()

    alive = live_target > 0.5
    team_balance = torch.where(home_target > 0.5,
                               team_logits.new_tensor(team_class_weights[1]),
                               team_logits.new_tensor(team_class_weights[0]))
    team_weight = frame_weights * team_balance * alive
    team_loss = (F.binary_cross_entropy_with_logits(
        team_logits, home_target, reduction="none") * team_weight).sum() \
        / team_weight.sum().clamp_min(1.0)
    return live_loss + team_loss


class _DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.dropout(self.activation(self.norm(self.conv(values))))
        return values + update


class TinyTemporalConvNet(nn.Module):
    """Compact non-causal dilated TCN with live/dead and conditional-team heads."""
    def __init__(self, input_channels: int, config: SequenceConfig = SequenceConfig()):
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        self.input_projection = nn.Conv1d(input_channels, config.hidden_channels, 1)
        self.blocks = nn.Sequential(*[
            _DilatedResidualBlock(config.hidden_channels, dilation, config.dropout)
            for dilation in config.dilations
        ])
        self.live_head = nn.Conv1d(config.hidden_channels, 1, 1)
        self.team_head = nn.Conv1d(config.hidden_channels, 1, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.blocks(self.input_projection(values))
        return self.live_head(hidden).squeeze(1), self.team_head(hidden).squeeze(1)


def set_reproducible(seed: int) -> None:
    """Seed all local RNGs and request deterministic PyTorch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_sequence_model(matches: tuple[PreparedMatch, ...], normalizer: FeatureNormalizer,
                         config: ExperimentConfig, device
                         ) -> tuple[TinyTemporalConvNet, list[dict]]:
    """Fit one fold using deterministic windows and on-the-fly team-swap augmentation."""
    set_reproducible(config.seed)
    target_device = torch.device(device)
    dataset = SequenceWindowDataset(
        matches, normalizer, config.window_frames, config.stride_frames,
        config.transition_radius_frames, config.transition_boost,
        config.segment_weighting, config.segment_weight_cap,
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
        pin_memory=target_device.type == "cuda", generator=loader_generator,
    )
    model = TinyTemporalConvNet(2 * len(normalizer.feature_names), config.sequence).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs * len(loader))
    )
    valid_labels = [match.labels[start:stop]
                    for match in matches for start, stop in match.valid_runs]
    valid_base_weights = [dataset.weights[index][start:stop]
                          for index, match in enumerate(matches)
                          for start, stop in match.valid_runs]
    live_weights, team_weights = training_loss_class_weights(
        valid_labels, valid_base_weights
    )
    swap_index = team_swap_channel_index(normalizer.feature_names)
    augmentation_generator = torch.Generator().manual_seed(config.seed + 1)

    history = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for values, labels, weights in loader:
            rows = torch.rand(values.shape[0], generator=augmentation_generator) < 0.5
            values, labels = swap_batch_teams(values, labels, swap_index, rows)
            values = values.to(target_device, non_blocking=True)
            labels = labels.to(target_device, non_blocking=True)
            weights = weights.to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            live_logits, team_logits = model(values)
            loss = sequence_loss(
                live_logits, team_logits, labels, weights, live_weights, team_weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / max(1, n_batches),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "n_batches": n_batches,
        })
    return model, history


@torch.no_grad()
def predict_prepared_match(model: nn.Module, match: PreparedMatch,
                           normalizer: FeatureNormalizer, device) -> tuple[np.ndarray, np.ndarray]:
    """Predict independent valid runs, with exact H/A swap ensembling."""
    if match.feature_names != normalizer.feature_names:
        raise ValueError("match and normalizer feature schemas differ")
    model.eval()
    p_alive = np.full(len(match.labels), 0.5, dtype=np.float64)
    p_home = np.full(len(match.labels), 0.5, dtype=np.float64)
    channel_index = torch.as_tensor(
        team_swap_channel_index(match.feature_names), dtype=torch.long, device=device
    )
    for start, stop in match.valid_runs:
        normalized = normalize_features(match.values[start:stop], normalizer)
        values = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(device)
        live, team = model(values)
        swapped_live, swapped_team = model(values[:, channel_index, :])
        alive = (torch.sigmoid(live) + torch.sigmoid(swapped_live)) * 0.5
        home = (torch.sigmoid(team) + 1.0 - torch.sigmoid(swapped_team)) * 0.5
        p_alive[start:stop] = alive[0].float().cpu().numpy()
        p_home[start:stop] = home[0].float().cpu().numpy()
    return p_alive, p_home


def cross_validate_sequence(cache_dir: str, work_dir: str, config: ExperimentConfig,
                            device, input_schema: str):
    """Four-fold match-grouped OOF over the non-reserved SkillCorner development set."""
    contract = input_cache_contract(input_schema)
    paths = _cache_paths(cache_dir)
    dev_ids = validate_exact_development_ids(paths)
    validate_development_ids(dev_ids)
    folds = development_folds(dev_ids, n_folds=4)
    print("preparing {} allowed external matches from explicit {} schema".format(
        len(dev_ids), contract.name
    ), flush=True)
    loaded = load_allowed_matches(cache_dir, dev_ids, contract)
    prepared = tuple(prepare_cached_match(cached, SKILLCORNER_FPS, CONTEXT_OFFSETS_S)
                     for cached in loaded)
    schema = prepared[0].feature_names
    if any(match.feature_names != schema for match in prepared):
        raise ValueError("prepared feature schema differs between matches")
    if len(schema) != contract.context_feature_count:
        raise ValueError("{} contextual feature width {} != expected {}".format(
            contract.name, len(schema), contract.context_feature_count
        ))
    by_id = {match.match_id: match for match in prepared}

    code_paths = {
        "evaluation/possession_metrics.py": os.path.join(ROOT, "evaluation", "possession_metrics.py"),
        "solution/external_labels.py": os.path.join(ROOT, "solution", "external_labels.py"),
        "solution/skillcorner_adapter.py": os.path.join(ROOT, "solution", "skillcorner_adapter.py"),
        "solution/temporal.py": os.path.join(ROOT, "solution", "temporal.py"),
        "solution/tracking_features.py": os.path.join(ROOT, "solution", "tracking_features.py"),
        "solution/train_external.py": os.path.join(ROOT, "solution", "train_external.py"),
        "solution/train_tcn_external.py": os.path.abspath(__file__),
    }
    if contract.name == "contact":
        code_paths.update({
            "solution/contact_cache.py": os.path.join(ROOT, "solution", "contact_cache.py"),
            "solution/contact_features.py": os.path.join(ROOT, "solution", "contact_features.py"),
        })
    provenance = build_run_provenance(
        {match_id: paths[match_id] for match_id in dev_ids},
        code_paths,
        schema,
        CONTEXT_OFFSETS_S,
        contract,
    )
    provenance["training_device"] = str(device)
    identity_payload = json.dumps({
        "config": asdict(config), "provenance": provenance,
    }, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(identity_payload.encode()).hexdigest()[:12]
    probability_dir = os.path.join(work_dir, "oof_tcn", run_id)
    os.makedirs(probability_dir, exist_ok=True)
    records, fold_reports = [], []
    parameter_count = 0
    for fold_number, validation_ids in enumerate(folds):
        train_ids = tuple(sorted(set(dev_ids) - set(validation_ids)))
        train_matches = tuple(by_id[match_id] for match_id in train_ids)
        normalizer = fit_fold_normalizer(train_matches)
        fold_config = replace(config, seed=config.seed + fold_number)
        print("fold {} train {} validate {}".format(
            fold_number, list(train_ids), list(validation_ids)), flush=True)
        model, history = train_sequence_model(
            train_matches, normalizer, fold_config, device
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        for epoch in history:
            print("  epoch {}/{} loss {:.6f} lr {:.7f}".format(
                epoch["epoch"], config.epochs, epoch["loss"], epoch["learning_rate"]
            ), flush=True)
        fold_reports.append({
            "fold": fold_number,
            "seed": fold_config.seed,
            "train_match_ids": list(train_ids),
            "validation_match_ids": list(validation_ids),
            "history": history,
        })
        for match_id in validation_ids:
            match = by_id[match_id]
            p_alive, p_home = predict_prepared_match(model, match, normalizer, device)
            output = os.path.join(probability_dir, "{}.npz".format(match_id))
            np.savez_compressed(
                output,
                match_id=np.array(match_id, dtype=np.int64),
                p_alive=p_alive,
                p_home=p_home,
                valid_runs=np.asarray(match.valid_runs, dtype=np.int64),
                input_schema=np.array(contract.name),
                feature_schema_version=np.array(contract.schema_version),
            )
            records.append({
                "match_id": match_id,
                "labels": match.labels,
                "valid_runs": match.valid_runs,
                "p_alive": p_alive,
                "p_home": p_home,
            })
            print("  predicted {}".format(match_id), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    probability_hashes = {
        str(record["match_id"]): _file_sha256(os.path.join(
            probability_dir, "{}.npz".format(record["match_id"])
        ))
        for record in records
    }
    provenance["oof_probability_sha256"] = probability_hashes
    provenance["oof_probability_directory"] = os.path.relpath(probability_dir, ROOT)
    return records, fold_reports, parameter_count, run_id, provenance


def baseline_comparison(work_dir: str, report_name: str, candidate_best: dict):
    """Load one named baseline and report a match-by-match paired spellF1 delta."""
    path = os.path.join(work_dir, "experiments", report_name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        baseline_best = json.load(handle)["best"]
    baseline_mean = float(baseline_best["mean_spell_f1"])
    candidate_mean = float(candidate_best["mean_spell_f1"])
    if not 0.0 <= baseline_mean <= 1.0 or not 0.0 <= candidate_mean <= 1.0:
        raise ValueError("spellF1 is outside [0, 1]")

    def by_match(best, label):
        pairs = {}
        for record in best["per_match"]:
            match_id = int(record["match_id"])
            if match_id in pairs:
                raise ValueError("{} repeats match {}".format(label, match_id))
            value = float(record["spell_f1"])
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} match {} spellF1 is outside [0, 1]".format(
                    label, match_id
                ))
            pairs[match_id] = value
        return pairs

    candidate = by_match(candidate_best, "candidate")
    baseline = by_match(baseline_best, "baseline")
    if set(candidate) != set(baseline):
        raise ValueError("candidate/baseline match IDs differ: candidate={} baseline={}".format(
            sorted(candidate), sorted(baseline)
        ))
    per_match = []
    for match_id in sorted(candidate):
        per_match.append({
            "match_id": match_id,
            "candidate_spell_f1": candidate[match_id],
            "baseline_spell_f1": baseline[match_id],
            "spell_f1_delta": round(candidate[match_id] - baseline[match_id], 15),
        })
    tolerance = 1e-15
    try:
        report_path = os.path.relpath(path, ROOT)
    except ValueError:
        report_path = os.path.abspath(path)
    return {
        "baseline_mean_spell_f1": baseline_mean,
        "candidate_mean_spell_f1": candidate_mean,
        "mean_spell_f1_delta": round(candidate_mean - baseline_mean, 15),
        "wins": sum(row["spell_f1_delta"] > tolerance for row in per_match),
        "ties": sum(abs(row["spell_f1_delta"]) <= tolerance for row in per_match),
        "losses": sum(row["spell_f1_delta"] < -tolerance for row in per_match),
        "per_match": per_match,
        "report_sha256": _file_sha256(path),
        "report_path": report_path,
    }


def run_experiment(cache_dir: str, work_dir: str, config: ExperimentConfig,
                   device, input_schema: str) -> tuple[str, dict]:
    """Train grouped OOF models, select decoder on OOF, and persist an exact JSON report."""
    started = time.time()
    contract = input_cache_contract(input_schema)
    records, folds, parameter_count, run_id, provenance = cross_validate_sequence(
        cache_dir, work_dir, config, device, contract.name
    )
    best_duration, decoder_reports = evaluate_decoder(records)
    baseline = baseline_comparison(
        work_dir, contract.baseline_report_name, decoder_reports[0]
    )
    result_payload = json.dumps({
        "run_id": run_id,
        "oof_probability_sha256": provenance["oof_probability_sha256"],
        "best_duration_config": asdict(best_duration),
        "best_metrics": {
            key: value for key, value in decoder_reports[0].items() if key != "per_match"
        },
    }, sort_keys=True, separators=(",", ":"))
    result_id = hashlib.sha256(result_payload.encode()).hexdigest()[:12]
    report = {
        "run_id": run_id,
        "result_id": result_id,
        "method": ("external-only bidirectional dilated TCN with live/team heads; "
                   "input_schema={}; segment_weighting={}".format(
                       contract.name, config.segment_weighting
                   )),
        "model_selection_data": "16 allowed SkillCorner development matches; grouped OOF only",
        "development_match_ids": sorted(int(record["match_id"]) for record in records),
        "excluded_reserved_match_ids": list(FINAL_TEST_MATCH_IDS),
        "config": asdict(config),
        "input_cache_contract": asdict(contract),
        "best_duration_config": asdict(best_duration),
        "best": decoder_reports[0],
        "all_decoder_configs": decoder_reports,
        "folds": folds,
        "parameter_count": parameter_count,
        "device": str(device),
        "torch_version": torch.__version__,
        "provenance": provenance,
        "baseline_comparison": baseline,
        "elapsed_seconds": time.time() - started,
    }
    experiment_dir = os.path.join(work_dir, "experiments")
    os.makedirs(experiment_dir, exist_ok=True)
    experiment_name = (contract.experiment_name if config.segment_weighting == "none"
                       else contract.experiment_name.replace(
                           "_v1", "_v2_segment_balanced"
                       ))
    path = os.path.join(
        experiment_dir, "{}_{}_{}.json".format(experiment_name, run_id, result_id)
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("best external OOF spell {:.4f}; wrote {}".format(
        report["best"]["mean_spell_f1"], path), flush=True)
    return path, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument(
        "--input-schema", choices=("base", "contact"), required=True,
        help="exact cache contract; never inferred from cache contents",
    )
    parser.add_argument("--work", default=DEFAULT_WORK)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--transition-radius", type=int, default=10)
    parser.add_argument("--transition-boost", type=float, default=4.0)
    parser.add_argument("--segment-weighting", choices=("none", "sqrt_inverse"),
                        default="none")
    parser.add_argument("--segment-weight-cap", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        window_frames=args.window,
        stride_frames=args.stride,
        learning_rate=args.learning_rate,
        transition_radius_frames=args.transition_radius,
        transition_boost=args.transition_boost,
        segment_weighting=args.segment_weighting,
        segment_weight_cap=args.segment_weight_cap,
        sequence=SequenceConfig(hidden_channels=args.hidden, dropout=args.dropout),
    )
    run_experiment(args.cache, args.work, config, args.device, args.input_schema)
    return 0


if __name__ == "__main__":
    sys.exit(main())
