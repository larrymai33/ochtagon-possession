"""Calibration-free probability composition and minimum-duration decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


H, A, D, X = 0, 1, 2, 3


@dataclass(frozen=True)
class DurationConfig:
    """Durations in seconds and transition costs in integrated log-probability units."""

    min_duration_s: tuple[float, float, float]
    team_switch_penalty: float
    dead_switch_penalty: float

    def __post_init__(self):
        if len(self.min_duration_s) != 3 or min(self.min_duration_s) < 0:
            raise ValueError("min_duration_s must contain three non-negative values")
        if self.team_switch_penalty < 0 or self.dead_switch_penalty < 0:
            raise ValueError("transition penalties must be non-negative")


def compose_emissions(p_alive: np.ndarray, p_home_given_alive: np.ndarray,
                      temperature: float = 1.0, epsilon: float = 1e-6) -> np.ndarray:
    """Compose coherent H/A/D log emissions from the two-stage classifiers."""
    alive = np.asarray(p_alive, dtype=np.float64)
    home = np.asarray(p_home_given_alive, dtype=np.float64)
    if alive.shape != home.shape or alive.ndim != 1:
        raise ValueError("probability arrays must be same-length vectors")
    if temperature <= 0 or epsilon <= 0 or epsilon >= 0.5:
        raise ValueError("temperature must be positive and epsilon must be in (0, .5)")
    alive = np.clip(alive, epsilon, 1.0 - epsilon)
    home = np.clip(home, epsilon, 1.0 - epsilon)
    return np.column_stack((
        np.log(alive) + np.log(home),
        np.log(alive) + np.log1p(-home),
        np.log1p(-alive),
    )) / temperature


def viterbi_min_duration(emissions: np.ndarray, mins: tuple[int, int, int],
                         penalties: np.ndarray) -> np.ndarray:
    """Exact three-label Viterbi decode with a chain of substates per label."""
    e = np.asarray(emissions, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != 3 or not len(e):
        raise ValueError("emissions must have non-empty shape [frames, 3]")
    if len(mins) != 3 or min(mins) < 1:
        raise ValueError("minimum durations must be three positive frame counts")
    penalties = np.asarray(penalties, dtype=np.float64)
    if penalties.shape != (3, 3):
        raise ValueError("penalties must have shape [3, 3]")

    offsets = np.concatenate(([0], np.cumsum(mins))).astype(int)
    n_states = int(offsets[-1])
    label_of = np.repeat(np.arange(3), mins)
    last = offsets[1:] - 1
    back = np.zeros((len(e), n_states), dtype=np.int32)
    score = np.full(n_states, -np.inf)
    score[offsets[:3]] = e[0]

    for time in range(1, len(e)):
        new = np.full(n_states, -np.inf)
        for label in range(3):
            offset, minimum = offsets[label], mins[label]
            choices = [
                (score[last[other]] - penalties[other, label], last[other])
                for other in range(3) if other != label
            ]
            best_value, best_state = max(choices)
            new[offset], back[time, offset] = best_value, best_state
            if minimum > 1:
                new[offset + 1:offset + minimum] = score[offset:offset + minimum - 1]
                back[time, offset + 1:offset + minimum] = np.arange(
                    offset, offset + minimum - 1
                )
            stay = score[offset + minimum - 1]
            if stay > new[offset + minimum - 1]:
                new[offset + minimum - 1] = stay
                back[time, offset + minimum - 1] = offset + minimum - 1
        score = new + e[time, label_of]

    path = np.empty(len(e), dtype=np.int8)
    state = int(np.argmax(score))
    for time in range(len(e) - 1, -1, -1):
        path[time] = label_of[state]
        state = int(back[time, state])
    return path


def decode_runs(emissions: np.ndarray, runs: tuple[tuple[int, int], ...], fps: float,
                config: DurationConfig) -> np.ndarray:
    """Decode valid runs independently and return X elsewhere."""
    e = np.asarray(emissions, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != 3:
        raise ValueError("emissions must have shape [frames, 3]")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    labels = np.full(len(e), X, dtype=np.int8)
    previous_stop = 0
    clean_runs = []
    for start, stop in runs:
        start, stop = int(start), int(stop)
        if start < previous_stop or start < 0 or stop <= start or stop > len(e):
            raise ValueError("runs must be ordered, disjoint, non-empty, and in bounds")
        clean_runs.append((start, stop))
        previous_stop = stop

    mins = tuple(max(1, int(round(seconds * fps))) for seconds in config.min_duration_s)
    penalties = np.array([
        [0.0, config.team_switch_penalty, config.dead_switch_penalty],
        [config.team_switch_penalty, 0.0, config.dead_switch_penalty],
        [config.dead_switch_penalty, config.dead_switch_penalty, 0.0],
    ])
    # Integrating evidence per second makes a fixed penalty portable between 10 Hz and 25 Hz.
    scaled = e / fps
    for start, stop in clean_runs:
        labels[start:stop] = viterbi_min_duration(scaled[start:stop], mins, penalties)
    return labels
