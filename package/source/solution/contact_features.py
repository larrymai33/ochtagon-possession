"""Provider-neutral, identity-free ball-contact features from raw tracking arrays."""

from __future__ import annotations

import numpy as np

from solution.tracking_features import FeatureBatch, TrackingSequence, valid_runs


CONTACT_FEATURE_SCHEMA_VERSION = "contact-v1"
TOP_K = 5
MOTION_K = 3
PRESSURE_RADII_METRES = (1, 2, 3, 5)


def _run_difference(values: np.ndarray, runs: tuple[tuple[int, int], ...],
                    fps: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    out = np.full(source.shape, np.nan, dtype=np.float32)
    for start, stop in runs:
        if stop - start > 1:
            out[start + 1:stop] = (source[start + 1:stop]
                                   - source[start:stop - 1]) * fps
    return out


def _masked_coordinates(sequence: TrackingSequence) -> tuple[np.ndarray, np.ndarray]:
    ball = np.asarray(sequence.ball_xy, dtype=np.float32).copy()
    players = np.asarray(sequence.player_xy, dtype=np.float32).copy()
    ball_ok = (np.asarray(sequence.ball_detected, dtype=bool)
               & np.isfinite(ball).all(axis=1))
    player_ok = (np.asarray(sequence.player_detected, dtype=bool)
                 & np.isfinite(players).all(axis=2))
    ball[~ball_ok] = np.nan
    players[~player_ok] = np.nan
    return ball, players


def extract_contact_features(sequence: TrackingSequence) -> FeatureBatch:
    """Extract per-team proximity and contest features without player identities."""
    runs = valid_runs(np.asarray(sequence.period))
    ball, players = _masked_coordinates(sequence)
    ball_velocity = _run_difference(ball, runs, sequence.fps)
    player_velocity = _run_difference(players, runs, sequence.fps)
    ball_speed = np.hypot(ball_velocity[:, 0], ball_velocity[:, 1])
    ball_accel = _run_difference(ball_speed, runs, sequence.fps)
    ball_velocity_delta = _run_difference(ball_velocity, runs, sequence.fps)
    ball_impulse = np.hypot(ball_velocity_delta[:, 0], ball_velocity_delta[:, 1])
    previous_ball_velocity = np.full(ball_velocity.shape, np.nan, dtype=np.float32)
    for start, stop in runs:
        if stop - start > 1:
            previous_ball_velocity[start + 1:stop] = ball_velocity[start:stop - 1]
    previous_speed = np.hypot(previous_ball_velocity[:, 0], previous_ball_velocity[:, 1])
    turn_denom = ball_speed * previous_speed
    ball_turn_cos = np.divide(
        np.sum(ball_velocity * previous_ball_velocity, axis=1),
        turn_denom,
        out=np.full(len(ball), np.nan, dtype=np.float32),
        where=turn_denom > 0,
    )

    ball_ok = np.isfinite(ball).all(axis=1)
    player_ok = np.isfinite(players).all(axis=2)
    raw_teams = np.asarray(sequence.player_team, dtype=np.int8)
    teams = (np.broadcast_to(raw_teams, player_ok.shape)
             if raw_teams.ndim == 1 else raw_teams)
    all_distances = np.hypot(players[:, :, 0] - ball[:, None, 0],
                             players[:, :, 1] - ball[:, None, 1])

    team_columns: dict[str, dict[str, np.ndarray]] = {}
    team_top: dict[str, np.ndarray] = {}
    for code, side in ((0, "home"), (1, "away")):
        distances = np.where(
            player_ok & (teams == code) & ball_ok[:, None], all_distances, np.inf
        )
        tie_x = players[:, :, 0]
        tie_y = players[:, :, 1]
        tie_vx = player_velocity[:, :, 0]
        tie_vy = player_velocity[:, :, 1]
        order = np.lexsort((tie_vy, tie_vx, tie_y, tie_x, distances), axis=1)
        ordered = np.take_along_axis(distances, order, axis=1)
        top = np.full((len(distances), TOP_K), np.nan, dtype=np.float32)
        available = min(TOP_K, ordered.shape[1])
        if available:
            top[:, :available] = ordered[:, :available]
        top[~np.isfinite(top)] = np.nan
        team_top[side] = top
        columns = {
            "{}_contact_d{}".format(side, rank + 1): top[:, rank]
            for rank in range(TOP_K)
        }
        for rank in range(TOP_K):
            columns["{}_contact_d{}_slope".format(side, rank + 1)] = _run_difference(
                top[:, rank], runs, sequence.fps
            )
        rows = np.arange(len(ball))
        for rank in range(MOTION_K):
            if rank < order.shape[1]:
                player_index = order[:, rank]
                ranked_player = players[rows, player_index].copy()
                ranked_velocity = player_velocity[rows, player_index].copy()
            else:
                ranked_player = np.full(ball.shape, np.nan, dtype=np.float32)
                ranked_velocity = np.full(ball.shape, np.nan, dtype=np.float32)
            have_player = np.isfinite(top[:, rank])
            ranked_player[~have_player] = np.nan
            ranked_velocity[~have_player] = np.nan
            relative_position = ball - ranked_player
            relative_velocity = ranked_velocity - ball_velocity
            distance_denom = np.hypot(relative_position[:, 0], relative_position[:, 1])
            closing = np.divide(
                np.sum(relative_position * relative_velocity, axis=1),
                distance_denom,
                out=np.full(len(ball), np.nan, dtype=np.float32),
                where=distance_denom > 0,
            )
            prefix = "{}_contact_d{}".format(side, rank + 1)
            columns[prefix + "_closing"] = closing
            ranked_speed = np.hypot(ranked_velocity[:, 0], ranked_velocity[:, 1])
            direction_denom = ranked_speed * ball_speed
            columns[prefix + "_direction_cos"] = np.divide(
                np.sum(ranked_velocity * ball_velocity, axis=1),
                direction_denom,
                out=np.full(len(ball), np.nan, dtype=np.float32),
                where=direction_denom > 0,
            )
            columns[prefix + "_relative_speed"] = np.hypot(
                relative_velocity[:, 0], relative_velocity[:, 1]
            )
        columns["{}_contact_impulse_coupling".format(side)] = np.divide(
            ball_impulse,
            1.0 + top[:, 0],
            out=np.full(len(ball), np.nan, dtype=np.float32),
            where=np.isfinite(top[:, 0]),
        )
        for radius in PRESSURE_RADII_METRES:
            count = (distances <= radius).sum(axis=1).astype(np.float32)
            count[~np.isfinite(top[:, 0])] = np.nan
            columns["{}_pressure_{}m".format(side, radius)] = count
        team_columns[side] = columns

    home_nearest = team_columns["home"]["home_contact_d1"]
    away_nearest = team_columns["away"]["away_contact_d1"]
    team_columns["home"]["home_contest_advantage"] = away_nearest - home_nearest
    team_columns["away"]["away_contest_advantage"] = home_nearest - away_nearest
    for rank in range(1, MOTION_K):
        home_distance = team_top["home"][:, rank]
        away_distance = team_top["away"][:, rank]
        suffix = "contest_d{}_advantage".format(rank + 1)
        team_columns["home"]["home_" + suffix] = away_distance - home_distance
        team_columns["away"]["away_" + suffix] = home_distance - away_distance
    for radius in PRESSURE_RADII_METRES:
        home_pressure = team_columns["home"]["home_pressure_{}m".format(radius)]
        away_pressure = team_columns["away"]["away_pressure_{}m".format(radius)]
        suffix = "pressure_{}m_advantage".format(radius)
        team_columns["home"]["home_" + suffix] = home_pressure - away_pressure
        team_columns["away"]["away_" + suffix] = away_pressure - home_pressure

    names: list[str] = []
    values: list[np.ndarray] = []
    for side in ("home", "away"):
        names.extend(team_columns[side])
        values.extend(team_columns[side].values())
    names.extend((
        "contact_ball_speed", "contact_ball_accel", "contact_ball_impulse",
        "contact_ball_turn_cos", "contact_contest_gap", "contact_contest_sum",
    ))
    values.extend((
        ball_speed, ball_accel, ball_impulse, ball_turn_cos,
        np.abs(home_nearest - away_nearest), home_nearest + away_nearest,
    ))
    for rank in range(1, MOTION_K):
        home_distance = team_top["home"][:, rank]
        away_distance = team_top["away"][:, rank]
        names.extend((
            "contact_contest_d{}_gap".format(rank + 1),
            "contact_contest_d{}_sum".format(rank + 1),
        ))
        values.extend((np.abs(home_distance - away_distance),
                       home_distance + away_distance))
    matrix = np.column_stack(values).astype(np.float32)
    return FeatureBatch(
        values=matrix,
        feature_names=tuple(names),
        valid_runs=runs,
        schema_version=CONTACT_FEATURE_SCHEMA_VERSION,
    )
