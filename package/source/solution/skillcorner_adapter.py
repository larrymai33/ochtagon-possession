"""Stream SkillCorner Open Data into the shared tracking and feature schemas."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os

import numpy as np

from solution.external_labels import (
    SKILLCORNER_SOURCE_COMMIT,
    build_phase_labels,
    load_match_metadata,
    load_phase_rows,
)
from solution.tracking_features import (
    FEATURE_SCHEMA_VERSION,
    FeatureBatch,
    TrackingSequence,
    extract_features,
)


SKILLCORNER_FPS = 10.0
ADAPTER_VERSION = "skillcorner-v1"


@dataclass(frozen=True)
class SkillCornerMatch:
    match_id: int
    player_ids: tuple[int, ...]
    sequence: TrackingSequence
    labels: np.ndarray


@dataclass(frozen=True)
class CachedMatch:
    match_id: int
    values: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    valid_runs: tuple[tuple[int, int], ...]
    source_commit: str
    schema_version: str


def _only_file(match_dir: str, suffix: str) -> str:
    paths = glob.glob(os.path.join(match_dir, "*" + suffix))
    if len(paths) != 1:
        raise ValueError("{}: expected one *{}, found {}".format(match_dir, suffix, len(paths)))
    return paths[0]


def _grow(array: np.ndarray, capacity: int, fill_value) -> np.ndarray:
    shape = (capacity,) + array.shape[1:]
    grown = np.full(shape, fill_value, dtype=array.dtype)
    grown[:len(array)] = array
    return grown


def _xy(record: dict) -> tuple[float, float]:
    x, y = record.get("x"), record.get("y")
    if x is None or y is None:
        return np.nan, np.nan
    return float(x), float(y)


def load_skillcorner_match(match_dir: str) -> SkillCornerMatch:
    """Load one match strictly; extrapolated coordinates remain available with flags false."""
    metadata_path = _only_file(match_dir, "_match.json")
    phases_path = _only_file(match_dir, "_phases_of_play.csv")
    tracking_path = _only_file(match_dir, "_tracking_extrapolated.jsonl")
    metadata = load_match_metadata(metadata_path)
    match_id = int(metadata["id"])

    home_id = int(metadata["home_team"]["id"])
    away_id = int(metadata["away_team"]["id"])
    player_to_team = {}
    for player in metadata.get("players", []):
        player_id, team_id = int(player["id"]), int(player["team_id"])
        if team_id == home_id:
            player_to_team[player_id] = 0
        elif team_id == away_id:
            player_to_team[player_id] = 1
    player_ids = tuple(sorted(player_to_team))
    if not player_ids:
        raise ValueError("{}: metadata has no home/away players".format(metadata_path))
    player_index = {player_id: index for index, player_id in enumerate(player_ids)}
    player_team = np.array([player_to_team[player_id] for player_id in player_ids], dtype=np.int8)

    capacity = max(int(period["end_frame"]) for period in metadata["match_periods"]) + 1
    capacity = max(capacity, 1)
    ball_xy = np.full((capacity, 2), np.nan, dtype=np.float32)
    ball_detected = np.zeros(capacity, dtype=bool)
    player_xy = np.full((capacity, len(player_ids), 2), np.nan, dtype=np.float32)
    player_detected = np.zeros((capacity, len(player_ids)), dtype=bool)
    period_values = np.zeros(capacity, dtype=np.int8)

    expected_frame = 0
    with open(tracking_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            frame = int(record["frame"])
            if frame != expected_frame:
                raise ValueError("{}:{}: tracking frames must be contiguous from 0; expected {}, "
                                 "got {}".format(
                                     tracking_path, line_number, expected_frame, frame))
            expected_frame += 1
            if frame >= capacity:
                new_capacity = max(frame + 1, capacity * 2)
                ball_xy = _grow(ball_xy, new_capacity, np.nan)
                ball_detected = _grow(ball_detected, new_capacity, False)
                player_xy = _grow(player_xy, new_capacity, np.nan)
                player_detected = _grow(player_detected, new_capacity, False)
                period_values = _grow(period_values, new_capacity, 0)
                capacity = new_capacity

            period = record.get("period")
            period_values[frame] = int(period) if period is not None else 0
            ball = record.get("ball_data") or {}
            ball_xy[frame] = _xy(ball)
            ball_detected[frame] = bool(ball.get("is_detected") or False)
            for player in record.get("player_data") or []:
                player_id = int(player["player_id"])
                if player_id not in player_index:
                    raise ValueError("{}:{}: player id {} is absent from match metadata".format(
                        tracking_path, line_number, player_id))
                index = player_index[player_id]
                player_xy[frame, index] = _xy(player)
                player_detected[frame, index] = bool(player.get("is_detected") or False)

    n_frames = expected_frame
    if not n_frames:
        raise ValueError("{}: tracking file is empty".format(tracking_path))
    ball_xy = ball_xy[:n_frames]
    ball_detected = ball_detected[:n_frames]
    player_xy = player_xy[:n_frames]
    player_detected = player_detected[:n_frames]
    period_values = period_values[:n_frames]
    labels = build_phase_labels(metadata, load_phase_rows(phases_path), n_frames=n_frames)
    labels[period_values == 0] = "X"

    sequence = TrackingSequence(
        ball_xy=ball_xy,
        ball_detected=ball_detected,
        player_xy=player_xy,
        player_detected=player_detected,
        player_team=player_team,
        period=period_values,
        fps=SKILLCORNER_FPS,
        pitch_m=(float(metadata["pitch_length"]), float(metadata["pitch_width"])),
    )
    return SkillCornerMatch(match_id, player_ids, sequence, labels)


def _cached(match: SkillCornerMatch, features: FeatureBatch) -> CachedMatch:
    return CachedMatch(
        match_id=match.match_id,
        values=features.values,
        labels=match.labels,
        feature_names=features.feature_names,
        valid_runs=features.valid_runs,
        source_commit=SKILLCORNER_SOURCE_COMMIT,
        schema_version=features.schema_version,
    )


def build_feature_cache(match_dir: str, path: str) -> CachedMatch:
    """Extract and persist one reproducible external-match feature cache."""
    match = load_skillcorner_match(match_dir)
    cached = _cached(match, extract_features(match.sequence))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        path,
        match_id=np.array(cached.match_id, dtype=np.int64),
        values=cached.values,
        labels=cached.labels,
        feature_names=np.array(cached.feature_names, dtype="<U64"),
        valid_runs=np.array(cached.valid_runs, dtype=np.int64),
        source_commit=np.array(cached.source_commit),
        schema_version=np.array(cached.schema_version),
        adapter_version=np.array(ADAPTER_VERSION),
    )
    return cached


def load_feature_cache(path: str, expected_source_commit: str = SKILLCORNER_SOURCE_COMMIT,
                       expected_schema: str = FEATURE_SCHEMA_VERSION) -> CachedMatch:
    """Load a cache only when its provenance and feature contract still match."""
    with np.load(path, allow_pickle=False) as archive:
        source_commit = str(archive["source_commit"].item())
        schema_version = str(archive["schema_version"].item())
        if source_commit != expected_source_commit:
            raise ValueError("source commit mismatch: cache {} != expected {}".format(
                source_commit, expected_source_commit))
        if schema_version != expected_schema:
            raise ValueError("feature schema mismatch: cache {} != expected {}".format(
                schema_version, expected_schema))
        runs_array = archive["valid_runs"]
        return CachedMatch(
            match_id=int(archive["match_id"].item()),
            values=archive["values"].copy(),
            labels=archive["labels"].copy(),
            feature_names=tuple(str(name) for name in archive["feature_names"].tolist()),
            valid_runs=tuple((int(start), int(stop)) for start, stop in runs_array.tolist()),
            source_commit=source_commit,
            schema_version=schema_version,
        )
