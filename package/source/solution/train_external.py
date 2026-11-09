"""Fit and cross-validate the two-stage model using external matches only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution.contact_cache import (  # noqa: E402
    COMBINED_FEATURE_SCHEMA_VERSION,
    load_combined_feature_cache,
)
from solution.external_labels import FINAL_TEST_MATCH_IDS  # noqa: E402
from solution.external_labels import development_folds, development_match_ids  # noqa: E402
from solution.model_artifact import ExternalArtifact, artifact_sha256, save_artifact  # noqa: E402
from solution.skillcorner_adapter import SKILLCORNER_FPS, load_feature_cache  # noqa: E402
from solution.temporal import DurationConfig, compose_emissions, decode_runs  # noqa: E402
from solution.tracking_features import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    add_temporal_context,
    swap_teams,
)
from evaluation import possession_metrics as pm  # noqa: E402


DEFAULT_CACHE = os.path.join(ROOT, "data", "cache", "external_model", "features")
DEFAULT_WORK = os.path.join(ROOT, "data", "cache", "external_model")
DEFAULT_ARTIFACT = os.path.join(DEFAULT_WORK, "frozen.joblib")


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.08
    max_iter: int = 120
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    random_state: int = 20261023


@dataclass(frozen=True)
class FittedModels:
    live_model: object
    team_model: object
    feature_names: tuple[str, ...]
    config: ModelConfig


BALL_DEPENDENT_TEAM_BASES = {
    "nearest_d", "second_d", "within_1m", "within_2m", "within_3m", "within_5m",
    "within_10m", "ball_centroid_d", "nearest_vx", "nearest_vy", "nearest_speed",
    "nearest_closing",
}


def source_visibility_view(values: np.ndarray, feature_names: tuple[str, ...]) -> np.ndarray:
    """Remove SkillCorner extrapolated-ball evidence when the source says it was not detected.

    BAR-ATH supplies no extrapolated coordinate on a missing-ball frame. Applying the same contract
    to external frames prevents the model from learning a provider-only shortcut.
    """
    out = np.array(values, dtype=np.float32, copy=True)
    positions = {name: index for index, name in enumerate(feature_names)}
    if "ball_detected" not in positions or "ball_available" not in positions:
        raise ValueError("feature schema needs ball_detected and ball_available flags")
    missing = out[:, positions["ball_detected"]] < 0.5
    out[missing, positions["ball_available"]] = 0.0
    for name, index in positions.items():
        dependent = (name.startswith("ball_")
                     and name not in ("ball_available", "ball_detected"))
        if name.startswith("home_") or name.startswith("away_"):
            dependent = name.split("_", 1)[1] in BALL_DEPENDENT_TEAM_BASES
        if dependent:
            out[missing, index] = np.nan
    return out


def balanced_sample_weights(labels: np.ndarray,
                            base_weights: np.ndarray | None = None) -> np.ndarray:
    """Weights whose total is equal for every class present.

    ``base_weights`` can first equalize temporal segments; class balancing is then applied
    without destroying those within-class relative weights.
    """
    values = np.asarray(labels)
    base = (np.ones(len(values), dtype=np.float64) if base_weights is None
            else np.asarray(base_weights, dtype=np.float64))
    if base.shape != (len(values),) or not np.isfinite(base).all() or np.any(base < 0):
        raise ValueError("base weights must be a finite non-negative label-length vector")
    classes = np.unique(values)
    if not len(classes):
        raise ValueError("cannot weight an empty label array")
    totals = {value: float(base[values == value].sum()) for value in classes.tolist()}
    if min(totals.values()) <= 0:
        raise ValueError("every class must have positive base-weight mass")
    total = float(base.sum())
    scale = {value: total / (len(classes) * totals[value]) for value in classes.tolist()}
    return base * np.array([scale[value] for value in values.tolist()], dtype=np.float64)


def _segments(labels: np.ndarray, runs) -> tuple[tuple[int, int], ...]:
    values = np.asarray(labels).astype("<U1")
    segments = []
    for start, stop in runs:
        start, stop = int(start), int(stop)
        if start < 0 or stop <= start or stop > len(values):
            raise ValueError("segment runs must be non-empty and in bounds")
        breaks = np.flatnonzero(values[start + 1:stop] != values[start:stop - 1]) + start + 1
        bounds = np.concatenate(([start], breaks, [stop]))
        segments.extend((int(bounds[i]), int(bounds[i + 1]))
                        for i in range(len(bounds) - 1))
    return tuple(segments)


def segment_sample_weights(labels: np.ndarray, runs, rows: np.ndarray) -> np.ndarray:
    """Give every sampled constant-label segment total mass one."""
    values = np.asarray(labels).astype("<U1")
    selected = np.asarray(rows, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(values)):
        raise ValueError("sample rows must be an in-bounds vector")
    weights = np.zeros(len(selected), dtype=np.float64)
    for start, stop in _segments(values, runs):
        positions = np.flatnonzero((selected >= start) & (selected < stop))
        if len(positions):
            weights[positions] = 1.0 / len(positions)
    if np.any(weights == 0):
        raise ValueError("every sample row must belong to a supplied valid run")
    return weights


def _sample_rows(labels: np.ndarray, runs, stride: int) -> np.ndarray:
    """Uniformly sample frames while guaranteeing every short segment one example."""
    values = np.asarray(labels).astype("<U1")
    uniform = np.flatnonzero(values != "X")[::stride]
    centres = np.array([(start + stop - 1) // 2 for start, stop in _segments(values, runs)],
                       dtype=np.int64)
    return np.unique(np.concatenate((uniform, centres)))


def augment_team_swap(values: np.ndarray, labels: np.ndarray,
                      feature_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    swapped_values, swapped_labels = swap_teams(values, feature_names, labels)
    return (np.concatenate((np.asarray(values), swapped_values), axis=0),
            np.concatenate((np.asarray(labels).astype("<U1"), swapped_labels), axis=0))


def validate_development_ids(match_ids) -> None:
    leaked = sorted(set(int(match_id) for match_id in match_ids) & set(FINAL_TEST_MATCH_IDS))
    if leaked:
        raise ValueError("final-test match IDs are forbidden in development: {}".format(leaked))


def _classifier(config: ModelConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        random_state=config.random_state,
        early_stopping=False,
    )


def fit_models(values: np.ndarray, labels: np.ndarray, feature_names: tuple[str, ...],
               config: ModelConfig = ModelConfig(),
               sample_weights: np.ndarray | None = None) -> FittedModels:
    """Fit live/dead and conditional-team models after exact team-swap augmentation."""
    x = np.asarray(values, dtype=np.float32)
    y = np.asarray(labels).astype("<U1")
    if x.ndim != 2 or len(x) != len(y) or x.shape[1] != len(feature_names):
        raise ValueError("training values/labels/feature_names have incompatible shapes")
    bad = sorted(set(y.tolist()) - {"H", "A", "D"})
    if bad:
        raise ValueError("training labels must be H/A/D, got {}".format(bad))
    base = (np.ones(len(y), dtype=np.float64) if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float64))
    if base.shape != (len(y),) or not np.isfinite(base).all() or np.any(base <= 0):
        raise ValueError("sample weights must be a finite positive row-length vector")
    x_aug, y_aug = augment_team_swap(x, y, tuple(feature_names))
    base_aug = np.concatenate((base, base))

    live_y = (y_aug != "D").astype(np.int8)
    live_model = _classifier(config)
    live_model.fit(x_aug, live_y,
                   sample_weight=balanced_sample_weights(live_y, base_aug))

    team_rows = y_aug != "D"
    team_y = (y_aug[team_rows] == "H").astype(np.int8)
    if len(np.unique(team_y)) != 2:
        raise ValueError("team training rows must contain both H and A")
    team_model = _classifier(config)
    team_model.fit(x_aug[team_rows], team_y,
                   sample_weight=balanced_sample_weights(team_y, base_aug[team_rows]))
    return FittedModels(live_model, team_model, tuple(feature_names), config)


def _positive_probability(model, values: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == 1)
    if len(positions) != 1:
        raise ValueError("binary model has no unique positive class 1")
    return model.predict_proba(values)[:, int(positions[0])]


def predict_probabilities(models, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float32)
    expected = tuple(models.feature_names)
    if x.ndim != 2 or x.shape[1] != len(expected):
        raise ValueError("inference feature width does not match trained schema")
    return (_positive_probability(models.live_model, x),
            _positive_probability(models.team_model, x))


def _cache_paths(cache_dir: str) -> dict[int, str]:
    paths = {}
    if not os.path.isdir(cache_dir):
        return paths
    for name in os.listdir(cache_dir):
        stem, extension = os.path.splitext(name)
        if extension == ".npz" and stem.isdigit():
            paths[int(stem)] = os.path.join(cache_dir, name)
    return paths


def _load_matches(cache_dir: str, match_ids) -> list:
    requested = tuple(int(match_id) for match_id in match_ids)
    validate_development_ids(requested)
    paths = _cache_paths(cache_dir)
    missing = sorted(set(requested) - set(paths))
    if missing:
        raise IOError("missing external feature caches: {}".format(missing))
    loaded = []
    for match_id in requested:
        path = paths[match_id]
        with np.load(path, allow_pickle=False) as archive:
            if "schema_version" not in archive.files:
                raise ValueError("cache manifest is missing schema_version: {}".format(path))
            schema = str(archive["schema_version"].item())
        if schema == FEATURE_SCHEMA_VERSION:
            cached = load_feature_cache(path)
        elif schema == COMBINED_FEATURE_SCHEMA_VERSION:
            cached = load_combined_feature_cache(path)
        else:
            raise ValueError("unsupported external cache schema {!r}: {}".format(
                schema, path
            ))
        if cached.match_id != match_id:
            raise ValueError("cache match id {} != requested {}: {}".format(
                cached.match_id, match_id, path
            ))
        loaded.append(cached)
    return loaded


def _training_arrays(caches, stride: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if stride < 1:
        raise ValueError("sample stride must be positive")
    if not caches:
        raise ValueError("at least one external cache is required")
    base_names = caches[0].feature_names
    context_names = None
    values, labels, weights = [], [], []
    for cached in caches:
        if cached.feature_names != base_names:
            raise ValueError("feature schema differs between external caches")
        visible = source_visibility_view(cached.values, base_names)
        contextual, names = add_temporal_context(
            visible, base_names, cached.valid_runs, SKILLCORNER_FPS
        )
        context_names = names if context_names is None else context_names
        if names != context_names:
            raise ValueError("temporal feature schema differs between external caches")
        rows = _sample_rows(cached.labels, cached.valid_runs, stride)
        values.append(contextual[rows])
        labels.append(cached.labels[rows])
        weights.append(segment_sample_weights(cached.labels, cached.valid_runs, rows))
    return (np.concatenate(values), np.concatenate(labels), np.concatenate(weights),
            context_names)


def _predict_match(models: FittedModels, cached) -> tuple[np.ndarray, np.ndarray]:
    visible = source_visibility_view(cached.values, cached.feature_names)
    values, names = add_temporal_context(
        visible, cached.feature_names, cached.valid_runs, SKILLCORNER_FPS
    )
    if names != tuple(models.feature_names):
        raise ValueError("inference temporal feature schema does not match trained schema")
    p_alive = np.full(len(values), 0.5, dtype=np.float64)
    p_home = np.full(len(values), 0.5, dtype=np.float64)
    rows = np.flatnonzero(cached.labels != "X")
    p_alive[rows], p_home[rows] = predict_probabilities(models, values[rows])
    return p_alive, p_home


def _codes(labels: np.ndarray) -> np.ndarray:
    out = np.full(len(labels), 3, dtype=np.int8)
    for code, name in enumerate(("H", "A", "D", "X")):
        out[np.asarray(labels) == name] = code
    return out


def decoder_candidates() -> tuple[DurationConfig, ...]:
    """Small predeclared grid; avoids turning autocorrelated frames into a giant search."""
    values = []
    for min_team in (0.1, 0.2, 0.4):
        for min_dead, team_penalty, dead_penalty in (
                (0.2, 0.0, 0.0),
                (0.5, 0.25, 0.25),
                (1.0, 0.5, 0.5),
                (2.0, 1.0, 2.0)):
            values.append(DurationConfig(
                (min_team, min_team, min_dead), team_penalty, dead_penalty
            ))
    return tuple(values)


def _metric_summary(gt_labels, pred_codes) -> dict:
    result = pm.evaluate(_codes(gt_labels), np.asarray(pred_codes, dtype=np.int8), bootstrap_n=0)
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "accuracy": result["accuracy"],
        "alive_dead_f1": result["alive_dead_f1"],
        "team_accuracy": result["team_acc_gt_alive"],
        "gt_spells": result["n_gt_spells"],
        "pred_spells": result["n_pred_spells"],
    }


def evaluate_decoder(oof_records: list[dict], candidates=None) -> tuple[DurationConfig, list[dict]]:
    candidates = tuple(candidates or decoder_candidates())
    reports = []
    for number, config in enumerate(candidates, start=1):
        per_match = []
        for record in oof_records:
            emissions = compose_emissions(record["p_alive"], record["p_home"])
            prediction = decode_runs(emissions, record["valid_runs"], 10.0, config)
            metrics = _metric_summary(record["labels"], prediction)
            metrics["match_id"] = record["match_id"]
            per_match.append(metrics)
        summary = {
            "config": asdict(config),
            "mean_spell_f1": float(np.mean([m["spell_f1"] for m in per_match])),
            "mean_macro_f1": float(np.mean([m["macro_f1"] for m in per_match])),
            "mean_accuracy": float(np.mean([m["accuracy"] for m in per_match])),
            "mean_alive_dead_f1": float(np.mean([m["alive_dead_f1"] for m in per_match])),
            "mean_team_accuracy": float(np.mean([m["team_accuracy"] for m in per_match])),
            "per_match": per_match,
        }
        reports.append(summary)
        print("decoder {}/{} spell {:.4f} macro {:.4f}  {}".format(
            number, len(candidates), summary["mean_spell_f1"], summary["mean_macro_f1"],
            asdict(config)), flush=True)
    reports.sort(key=lambda report: (
        -report["mean_spell_f1"], -report["mean_macro_f1"],
        report["config"]["team_switch_penalty"] + report["config"]["dead_switch_penalty"],
    ))
    return DurationConfig(**reports[0]["config"]), reports


def cross_validate(cache_dir: str, work_dir: str, config: ModelConfig,
                   stride: int = 5, n_folds: int = 4) -> tuple[list[dict], tuple[str, ...]]:
    paths = _cache_paths(cache_dir)
    dev_ids = development_match_ids(paths)
    validate_development_ids(dev_ids)
    folds = development_folds(dev_ids, n_folds=n_folds)
    config_text = json.dumps(asdict(config), sort_keys=True)
    run_id = hashlib.sha256((config_text + ":stride={}:temporal-context-v1:segment-balanced-v1".format(
        stride)).encode()).hexdigest()[:12]
    probability_dir = os.path.join(work_dir, "oof", run_id)
    os.makedirs(probability_dir, exist_ok=True)

    records = []
    schema = None
    for fold_number, validation_ids in enumerate(folds):
        train_ids = tuple(sorted(set(dev_ids) - set(validation_ids)))
        print("fold {} train {} validate {}".format(
            fold_number, list(train_ids), list(validation_ids)), flush=True)
        train_caches = _load_matches(cache_dir, train_ids)
        train_x, train_y, train_weight, names = _training_arrays(train_caches, stride)
        schema = names if schema is None else schema
        if names != schema:
            raise ValueError("feature schema changed between folds")
        print("  fitting {} sampled rows (+ team-swap augmentation)".format(len(train_y)),
              flush=True)
        models = fit_models(train_x, train_y, names, config, sample_weights=train_weight)
        for cached in _load_matches(cache_dir, validation_ids):
            p_alive, p_home = _predict_match(models, cached)
            output = os.path.join(probability_dir, "{}.npz".format(cached.match_id))
            np.savez_compressed(output, p_alive=p_alive, p_home=p_home)
            records.append({
                "match_id": cached.match_id,
                "labels": cached.labels,
                "valid_runs": cached.valid_runs,
                "p_alive": p_alive,
                "p_home": p_home,
            })
            print("  predicted {}".format(cached.match_id), flush=True)
    return records, schema


def fit_final(cache_dir: str, model_path: str, config: ModelConfig,
              duration_config: DurationConfig, feature_names: tuple[str, ...], stride: int,
              cv_report: dict) -> ExternalArtifact:
    paths = _cache_paths(cache_dir)
    dev_ids = development_match_ids(paths)
    validate_development_ids(dev_ids)
    caches = _load_matches(cache_dir, dev_ids)
    input_schemas = {cached.schema_version for cached in caches}
    if len(input_schemas) != 1:
        raise ValueError("final training caches have mixed input schemas: {}".format(
            sorted(input_schemas)
        ))
    input_schema = next(iter(input_schemas))
    values, labels, weights, names = _training_arrays(caches, stride)
    if names != feature_names:
        raise ValueError("final feature schema differs from cross-validation")
    models = fit_models(values, labels, names, config, sample_weights=weights)
    artifact = ExternalArtifact(
        live_model=models.live_model,
        team_model=models.team_model,
        feature_names=names,
        duration_config=duration_config,
        temperature=1.0,
        manifest={
            "model_config": asdict(config),
            "duration_config": asdict(duration_config),
            "sample_stride": stride,
            "sample_weighting": "equal-mass-per-label-segment-then-class-balanced",
            "input_feature_schema": input_schema,
            "development_match_ids": list(dev_ids),
            "excluded_final_test_ids": list(FINAL_TEST_MATCH_IDS),
            "cross_validation": cv_report,
        },
    )
    save_artifact(model_path, artifact)
    print("wrote {} sha256 {}".format(model_path, artifact_sha256(model_path)), flush=True)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--work", default=DEFAULT_WORK)
    parser.add_argument("--model", default=DEFAULT_ARTIFACT)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--fit-final", action="store_true")
    args = parser.parse_args()
    config = ModelConfig(max_iter=args.max_iter, max_leaf_nodes=args.max_leaf_nodes,
                         min_samples_leaf=args.min_samples_leaf)
    records, names = cross_validate(args.cache, args.work, config, args.stride)
    best, reports = evaluate_decoder(records)
    report = {
        "model_config": asdict(config),
        "sample_stride": args.stride,
        "sample_weighting": "equal-mass-per-label-segment-then-class-balanced",
        "best": reports[0],
        "all_decoder_configs": reports,
    }
    experiment_dir = os.path.join(args.work, "experiments")
    os.makedirs(experiment_dir, exist_ok=True)
    report_path = os.path.join(experiment_dir, "hgb_tracking_segment_balanced_v3.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("best external CV spell {:.4f}; wrote {}".format(
        reports[0]["mean_spell_f1"], report_path), flush=True)
    if args.fit_final:
        fit_final(args.cache, args.model, config, best, names, args.stride, reports[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
