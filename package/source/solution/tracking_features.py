"""Provider-neutral tracking schema and permutation-invariant physical features."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


FEATURE_SCHEMA_VERSION = "tracking-v1"
RADIUS_METRES = (1, 2, 3, 5, 10)
TEAM_FEATURE_BASES = (
    "n_players", "n_detected", "nearest_d", "second_d",
    "within_1m", "within_2m", "within_3m", "within_5m", "within_10m",
    "speed_mean", "speed_median", "speed_q75",
    "centroid_x_norm", "centroid_y_norm", "spread_x", "spread_y",
    "compactness", "ball_centroid_d",
    "nearest_vx", "nearest_vy", "nearest_speed", "nearest_closing",
)
TEMPORAL_CONTEXT_FEATURES = (
    "ball_detected", "ball_line_d", "ball_vx", "ball_vy", "ball_speed", "ball_accel",
    "n_detected_players",
    "home_nearest_d", "away_nearest_d",
    "home_second_d", "away_second_d",
    "home_within_2m", "away_within_2m",
    "home_within_5m", "away_within_5m",
    "home_speed_mean", "away_speed_mean",
    "home_ball_centroid_d", "away_ball_centroid_d",
    "home_nearest_closing", "away_nearest_closing",
)


@dataclass(frozen=True)
class TrackingSequence:
    """Metric tracking for one match; coordinates use a pitch-centred origin."""

    ball_xy: np.ndarray
    ball_detected: np.ndarray
    player_xy: np.ndarray
    player_detected: np.ndarray
    player_team: np.ndarray
    period: np.ndarray
    fps: float
    pitch_m: tuple[float, float]

    def __post_init__(self):
        t = len(self.ball_xy)
        if np.shape(self.ball_xy) != (t, 2):
            raise ValueError("ball_xy must have shape [frames, 2]")
        if np.shape(self.ball_detected) != (t,):
            raise ValueError("ball_detected must have shape [frames]")
        if np.ndim(self.player_xy) != 3 or np.shape(self.player_xy)[0] != t \
                or np.shape(self.player_xy)[2] != 2:
            raise ValueError("player_xy must have shape [frames, players, 2]")
        n_players = np.shape(self.player_xy)[1]
        if np.shape(self.player_detected) != (t, n_players):
            raise ValueError("player_detected must have shape [frames, players]")
        if np.shape(self.player_team) not in ((n_players,), (t, n_players)):
            raise ValueError("player_team must have shape [players] or [frames, players]")
        if np.shape(self.period) != (t,):
            raise ValueError("period must have shape [frames]")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        if len(self.pitch_m) != 2 or min(self.pitch_m) <= 0:
            raise ValueError("pitch_m must be (positive length, positive width)")


@dataclass(frozen=True)
class FeatureBatch:
    values: np.ndarray
    feature_names: tuple[str, ...]
    valid_runs: tuple[tuple[int, int], ...]
    schema_version: str = FEATURE_SCHEMA_VERSION


def valid_runs(period: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Half-open constant non-zero period runs."""
    period = np.asarray(period)
    runs = []
    start = None
    current = 0
    for index, value in enumerate(period.tolist() + [0]):
        value = int(value)
        if start is not None and value != current:
            runs.append((start, index))
            start = None
        if start is None and value != 0:
            start, current = index, value
    return tuple(runs)


def _run_difference(values: np.ndarray, runs: tuple[tuple[int, int], ...],
                    fps: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    out = np.full(source.shape, np.nan, dtype=np.float64)
    for start, stop in runs:
        if stop - start > 1:
            out[start + 1:stop] = (source[start + 1:stop] - source[start:stop - 1]) * fps
    return out


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.where(mask, values, 0.0)
    count = mask.sum(axis=1)
    total = selected.sum(axis=1)
    return np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)


def _nan_reduction(values: np.ndarray, operation: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if operation == "median":
            return np.nanmedian(values, axis=1)
        if operation == "q75":
            return np.nanpercentile(values, 75, axis=1)
    raise ValueError("unknown reduction {}".format(operation))


def _team_features(team: int, name: str, sequence: TrackingSequence,
                   teams: np.ndarray, player_velocity: np.ndarray,
                   ball_velocity: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    xy = np.asarray(sequence.player_xy, dtype=np.float64)
    ball = np.asarray(sequence.ball_xy, dtype=np.float64)
    finite_player = np.isfinite(xy).all(axis=2)
    member = (teams == team) & finite_player
    detected = member & np.asarray(sequence.player_detected, dtype=bool)
    count = member.sum(axis=1).astype(float)
    detected_count = detected.sum(axis=1).astype(float)

    ball_available = np.isfinite(ball).all(axis=1)
    distance = np.hypot(xy[:, :, 0] - ball[:, None, 0],
                        xy[:, :, 1] - ball[:, None, 1])
    usable_distance = np.where(member & ball_available[:, None], distance, np.inf)
    ordered = np.sort(usable_distance, axis=1)
    nearest = ordered[:, 0]
    nearest[~np.isfinite(nearest)] = np.nan
    if ordered.shape[1] > 1:
        second = ordered[:, 1]
        second[~np.isfinite(second)] = np.nan
    else:
        second = np.full(len(ball), np.nan)

    columns = [count, detected_count, nearest, second]
    names = [
        "{}_n_players".format(name), "{}_n_detected".format(name),
        "{}_nearest_d".format(name), "{}_second_d".format(name),
    ]
    for radius in RADIUS_METRES:
        within = (usable_distance <= radius).sum(axis=1).astype(float)
        within[~ball_available] = np.nan
        names.append("{}_within_{}m".format(name, radius))
        columns.append(within)

    player_speed = np.hypot(player_velocity[:, :, 0], player_velocity[:, :, 1])
    speed_values = np.where(member, player_speed, np.nan)
    speed_mean = _masked_mean(player_speed, member & np.isfinite(player_speed))
    speed_median = _nan_reduction(speed_values, "median")
    speed_q75 = _nan_reduction(speed_values, "q75")
    names.extend(("{}_speed_mean".format(name), "{}_speed_median".format(name),
                  "{}_speed_q75".format(name)))
    columns.extend((speed_mean, speed_median, speed_q75))

    cx = _masked_mean(xy[:, :, 0], member)
    cy = _masked_mean(xy[:, :, 1], member)
    dx = xy[:, :, 0] - cx[:, None]
    dy = xy[:, :, 1] - cy[:, None]
    spread_x = np.sqrt(_masked_mean(dx * dx, member))
    spread_y = np.sqrt(_masked_mean(dy * dy, member))
    compactness = np.sqrt(_masked_mean(dx * dx + dy * dy, member))
    ball_centroid_d = np.hypot(ball[:, 0] - cx, ball[:, 1] - cy)
    ball_centroid_d[~ball_available] = np.nan
    length, width = sequence.pitch_m
    names.extend((
        "{}_centroid_x_norm".format(name), "{}_centroid_y_norm".format(name),
        "{}_spread_x".format(name), "{}_spread_y".format(name),
        "{}_compactness".format(name), "{}_ball_centroid_d".format(name),
    ))
    columns.extend((cx / (length / 2.0), cy / (width / 2.0), spread_x, spread_y,
                    compactness, ball_centroid_d))

    nearest_index = np.argmin(usable_distance, axis=1)
    rows = np.arange(len(ball))
    nearest_velocity = player_velocity[rows, nearest_index].copy()
    have_nearest = np.isfinite(nearest)
    nearest_velocity[~have_nearest] = np.nan
    nearest_speed = np.hypot(nearest_velocity[:, 0], nearest_velocity[:, 1])
    relative_position = ball - xy[rows, nearest_index]
    relative_velocity = nearest_velocity - ball_velocity
    denom = np.hypot(relative_position[:, 0], relative_position[:, 1])
    closing = np.divide(
        relative_position[:, 0] * relative_velocity[:, 0]
        + relative_position[:, 1] * relative_velocity[:, 1],
        denom,
        out=np.full(len(ball), np.nan),
        where=(denom > 0) & have_nearest,
    )
    names.extend(("{}_nearest_vx".format(name), "{}_nearest_vy".format(name),
                  "{}_nearest_speed".format(name), "{}_nearest_closing".format(name)))
    columns.extend((nearest_velocity[:, 0], nearest_velocity[:, 1], nearest_speed, closing))
    return names, columns


def extract_features(sequence: TrackingSequence) -> FeatureBatch:
    """Create one stable feature matrix shared by external and target adapters."""
    ball = np.asarray(sequence.ball_xy, dtype=np.float64)
    player = np.asarray(sequence.player_xy, dtype=np.float64)
    runs = valid_runs(np.asarray(sequence.period))
    ball_velocity = _run_difference(ball, runs, sequence.fps)
    player_velocity = _run_difference(player, runs, sequence.fps)
    ball_speed = np.hypot(ball_velocity[:, 0], ball_velocity[:, 1])
    ball_accel = _run_difference(ball_speed, runs, sequence.fps)
    ball_available = np.isfinite(ball).all(axis=1)
    length, width = sequence.pitch_m
    line_distance = np.fmin(length / 2.0 - np.abs(ball[:, 0]),
                            width / 2.0 - np.abs(ball[:, 1]))
    line_distance[~ball_available] = np.nan

    finite_player = np.isfinite(player).all(axis=2)
    names = [
        "ball_available", "ball_detected", "ball_x_norm", "ball_y_norm",
        "ball_line_d", "ball_vx", "ball_vy", "ball_speed", "ball_accel",
        "n_players", "n_detected_players",
    ]
    columns = [
        ball_available.astype(float), np.asarray(sequence.ball_detected, dtype=float),
        ball[:, 0] / (length / 2.0), ball[:, 1] / (width / 2.0), line_distance,
        ball_velocity[:, 0], ball_velocity[:, 1], ball_speed, ball_accel,
        finite_player.sum(axis=1).astype(float),
        (finite_player & np.asarray(sequence.player_detected, dtype=bool)).sum(axis=1).astype(float),
    ]

    raw_teams = np.asarray(sequence.player_team, dtype=np.int8)
    teams = (np.broadcast_to(raw_teams, finite_player.shape)
             if raw_teams.ndim == 1 else raw_teams)
    for code, side in ((0, "home"), (1, "away")):
        team_names, team_columns = _team_features(
            code, side, sequence, teams, player_velocity, ball_velocity
        )
        names.extend(team_names)
        columns.extend(team_columns)

    values = np.column_stack(columns).astype(np.float32)
    return FeatureBatch(values=values, feature_names=tuple(names), valid_runs=runs)


def _offset_name(seconds: float) -> str:
    text = ("{:.3f}".format(seconds)).rstrip("0").rstrip(".").replace(".", "p")
    return text + "s"


def add_temporal_context(values: np.ndarray, feature_names: tuple[str, ...],
                         runs: tuple[tuple[int, int], ...], fps: float,
                         offsets_s: tuple[float, ...] = (0.3, 1.0, 3.0)
                         ) -> tuple[np.ndarray, tuple[str, ...]]:
    """Append centred past/future deltas and means without crossing a period run."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != len(feature_names):
        raise ValueError("values and feature_names have incompatible shapes")
    if fps <= 0 or any(offset <= 0 for offset in offsets_s):
        raise ValueError("fps and temporal offsets must be positive")
    positions = {name: index for index, name in enumerate(feature_names)}
    selected = [name for name in TEMPORAL_CONTEXT_FEATURES if name in positions]
    # Allow focused synthetic/alternate schemas while keeping the production set explicit.
    if not selected:
        selected = [name for name in feature_names
                    if name.startswith("home_") or name.startswith("away_")]

    columns = [source]
    names = list(feature_names)
    for seconds in offsets_s:
        offset = max(1, int(round(seconds * fps)))
        label = _offset_name(seconds)
        for name in selected:
            current = source[:, positions[name]].astype(np.float64)
            past = np.full(len(source), np.nan)
            future = np.full(len(source), np.nan)
            for start, stop in runs:
                if stop - start > offset:
                    past[start + offset:stop] = current[start:stop - offset]
                    future[start:stop - offset] = current[start + offset:stop]
            columns.append((future - past)[:, None])
            names.append("{}_delta_{}".format(name, label))
            columns.append(((future + past) / 2.0)[:, None])
            names.append("{}_mean_{}".format(name, label))
    return np.concatenate(columns, axis=1).astype(np.float32), tuple(names)


def swap_teams(values: np.ndarray, feature_names: tuple[str, ...],
               labels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Swap explicit home/away feature pairs and H/A labels for symmetry augmentation."""
    swapped = np.array(values, copy=True)
    positions = {name: index for index, name in enumerate(feature_names)}
    home_names = [name for name in feature_names if name.startswith("home_")]
    away_names = [name for name in feature_names if name.startswith("away_")]
    for home in home_names:
        away = "away_" + home[len("home_"):]
        if away not in positions:
            raise ValueError("feature schema is missing team pair for {}".format(home))
        hi, ai = positions[home], positions[away]
        swapped[:, hi] = values[:, ai]
        swapped[:, ai] = values[:, hi]
    orphaned = [name for name in away_names
                if "home_" + name[len("away_"):] not in positions]
    if orphaned:
        raise ValueError("feature schema has unpaired away features: {}".format(orphaned))

    swapped_labels = None
    if labels is not None:
        swapped_labels = np.asarray(labels).astype("<U1", copy=True)
        original = np.asarray(labels).astype("<U1")
        swapped_labels[original == "H"] = "A"
        swapped_labels[original == "A"] = "H"
    return swapped, swapped_labels
