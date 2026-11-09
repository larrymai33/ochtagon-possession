"""Construct rule-compliant H/A/D labels from other-match SkillCorner phases."""

from __future__ import annotations

import csv
import json
from typing import Iterable, Mapping, Sequence

import numpy as np


SKILLCORNER_SOURCE_URL = "https://github.com/SkillCorner/opendata.git"
SKILLCORNER_SOURCE_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
FINAL_TEST_MATCH_IDS = (1927964, 1996436, 2010085, 2017461)


def load_match_metadata(path: str) -> dict:
    """Load the fields needed for labels/features without changing provider values."""
    with open(path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    for key in ("home_team", "away_team", "match_periods"):
        if key not in metadata:
            raise ValueError("{}: metadata has no {!r}".format(path, key))
    return metadata


def load_phase_rows(path: str) -> list[dict]:
    """Read phase bounds/team IDs, converting the provider CSV's numeric strings."""
    integer_fields = (
        "index", "match_id", "frame_start", "frame_end", "period",
        "team_in_possession_id",
    )
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), start=2):
            row = dict(raw)
            try:
                for field in integer_fields:
                    row[field] = int(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("{}:{}: invalid phase row: {}".format(
                    path, line_number, error))
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _as_int(row: Mapping, field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError("phase row has invalid {!r}: {!r}".format(field, row.get(field)))


def build_phase_labels(metadata: Mapping, phase_rows: Iterable[Mapping],
                       n_frames: int | None = None) -> np.ndarray:
    """Return provider phase labels, using D only for uncovered in-period frames.

    Metadata period ends are inclusive. Phase ends are exclusive. A phase which extends slightly
    into a provider prefill/gap is clipped to its declared period, matching the tracking rows.
    """
    periods = {}
    for raw in metadata["match_periods"]:
        number = int(raw["period"])
        start, stop = int(raw["start_frame"]), int(raw["end_frame"]) + 1
        if start < 0 or stop <= start:
            raise ValueError("invalid metadata period {} bounds {}..{}".format(
                number, start, stop - 1))
        if number in periods:
            raise ValueError("duplicate metadata period {}".format(number))
        periods[number] = (start, stop)
    if not periods:
        raise ValueError("metadata has no match periods")

    if n_frames is None:
        n_frames = max(stop for _, stop in periods.values())
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")

    labels = np.full(n_frames, "X", dtype="<U1")
    occupied = np.zeros(n_frames, dtype=bool)
    for start, stop in periods.values():
        labels[max(0, start):min(n_frames, stop)] = "D"

    home_id = int(metadata["home_team"]["id"])
    away_id = int(metadata["away_team"]["id"])
    side_for_id = {home_id: "H", away_id: "A"}

    for row in phase_rows:
        start = _as_int(row, "frame_start")
        stop = _as_int(row, "frame_end")
        period = _as_int(row, "period")
        team_id = _as_int(row, "team_in_possession_id")
        if stop <= start:
            raise ValueError("phase frame_start {} must be before frame_end {}".format(
                start, stop))
        if period not in periods:
            raise ValueError("phase declares unknown period {}".format(period))
        if team_id not in side_for_id:
            raise ValueError("unknown possession team id {}".format(team_id))

        period_start, period_stop = periods[period]
        low = max(0, start, period_start)
        high = min(n_frames, stop, period_stop)
        if high <= low:
            raise ValueError("phase {}..{} is outside period {}".format(start, stop, period))
        if occupied[low:high].any():
            raise ValueError("phase {}..{} overlaps another phase".format(start, stop))
        labels[low:high] = side_for_id[team_id]
        occupied[low:high] = True
    return labels


def development_match_ids(all_ids: Iterable[int]) -> tuple[int, ...]:
    """Sorted non-holdout IDs; never return a final-test match."""
    final = set(FINAL_TEST_MATCH_IDS)
    return tuple(sorted({int(match_id) for match_id in all_ids} - final))


def development_folds(match_ids: Sequence[int], n_folds: int = 4) -> tuple[tuple[int, ...], ...]:
    """Deterministic round-robin match folds for grouped cross-validation."""
    ids = tuple(sorted(int(match_id) for match_id in match_ids))
    leaked = sorted(set(ids) & set(FINAL_TEST_MATCH_IDS))
    if leaked:
        raise ValueError("final-test match IDs are not allowed in development folds: {}".format(
            leaked))
    if n_folds < 2 or n_folds > len(ids):
        raise ValueError("n_folds must be between 2 and the number of matches")
    return tuple(tuple(ids[index::n_folds]) for index in range(n_folds))
