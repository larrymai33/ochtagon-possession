"""Strict BAR-ATH EPTS adapter for the provider-neutral tracking schema."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from solution.tracking_features import TrackingSequence


def _slot_teams(meta_path: str) -> np.ndarray:
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2 or any(team_id is None for team_id in team_ids):
        raise ValueError("{}: expected exactly two metadata teams".format(meta_path))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: metadata has no Player slots".format(meta_path))
    unknown = sorted({team_id for team_id in slots if team_id not in team_ids})
    if unknown:
        raise ValueError("{}: player slots reference unknown teams {}".format(meta_path, unknown))
    return np.array([0 if team_id == team_ids[0] else 1 for team_id in slots], dtype=np.int8)


def _triplet(group: str, path: str, line_number: int) -> tuple[float, float, float]:
    fields = group.split(",")
    if len(fields) != 3:
        raise ValueError("{}:{}: expected x,y,speed triplet".format(path, line_number))
    try:
        return float(fields[0]), float(fields[1]), float(fields[2])
    except ValueError as error:
        raise ValueError("{}:{}: invalid numeric triplet: {}".format(path, line_number, error))


def load_target_sequence(meta_path: str, raw_path: str, n_frames: int,
                         pitch_m: tuple[float, float],
                         runs: tuple[tuple[int, int], ...], fps: float) -> TrackingSequence:
    """Parse exactly one EPTS row per requested frame; never skip malformed data."""
    teams = _slot_teams(meta_path)
    n_slots = len(teams)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    player_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball_detected = np.zeros(n_frames, dtype=bool)
    player_detected = np.zeros((n_frames, n_slots), dtype=bool)
    length, width = float(pitch_m[0]), float(pitch_m[1])

    count = 0
    with open(raw_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split(":")
            if len(parts) != 3:
                raise ValueError("{}:{}: expected frame:players:ball".format(
                    raw_path, line_number))
            try:
                frame_id = int(parts[0])
            except ValueError:
                raise ValueError("{}:{}: frame_id is not an integer".format(
                    raw_path, line_number))
            expected = count + 1
            if frame_id != expected:
                raise ValueError("{}:{}: expected frame_id {}, got {}".format(
                    raw_path, line_number, expected, frame_id))
            if count >= n_frames:
                raise ValueError("{}: more than {} tracking rows".format(raw_path, n_frames))

            player_groups = parts[1].split(";")
            if len(player_groups) != n_slots:
                raise ValueError("{}:{}: expected {} player slots, got {}".format(
                    raw_path, line_number, n_slots, len(player_groups)))
            for slot, group in enumerate(player_groups):
                x, y, _ = _triplet(group, raw_path, line_number)
                if np.isfinite(x) and np.isfinite(y):
                    player_xy[count, slot] = (x * length - length / 2.0,
                                              y * width - width / 2.0)
                    player_detected[count, slot] = True
            x, y, _ = _triplet(parts[2], raw_path, line_number)
            if np.isfinite(x) and np.isfinite(y):
                ball_xy[count] = (x * length - length / 2.0,
                                  y * width - width / 2.0)
                ball_detected[count] = True
            count += 1
    if count != n_frames:
        raise ValueError("{}: expected {} tracking rows, got {}".format(
            raw_path, n_frames, count))

    period = np.zeros(n_frames, dtype=np.int8)
    previous_stop = 0
    for number, (start, stop) in enumerate(runs, start=1):
        if start < previous_stop or start < 0 or stop <= start or stop > n_frames:
            raise ValueError("invalid/non-ordered target run {}..{}".format(start, stop))
        period[start:stop] = number
        previous_stop = stop

    return TrackingSequence(
        ball_xy=ball_xy,
        ball_detected=ball_detected,
        player_xy=player_xy,
        player_detected=player_detected,
        player_team=teams,
        period=period,
        fps=float(fps),
        pitch_m=(length, width),
    )


def load_bar_ath_sequence(data_dir: str | None = None) -> TrackingSequence:
    """Load the real target using only published data contracts and tracking assets."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from games import GAME, GAMES, half_video_blocks, require, source_frames

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    runs = tuple((first, last + 1) for _, first, last in half_video_blocks(GAME))
    return load_target_sequence(
        meta_path,
        raw_path,
        n_frames=source_frames("tactical", GAME),
        pitch_m=GAMES[GAME]["pitch"],
        runs=runs,
        fps=GAMES[GAME]["fps"],
    )
