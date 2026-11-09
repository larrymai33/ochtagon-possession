"""Pure, run-local time-grid conversion for the frozen tracking ensemble.

Tracking is sampled on a physical-time grid before any features are built.  Each
nonzero period run has its own anchor, and concatenated sampled runs receive
distinct period IDs.  Interpolation uses only the two adjacent native frames;
it never searches past a missing coordinate for a replacement observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solution.tracking_features import TrackingSequence, valid_runs


@dataclass(frozen=True)
class ResampledTracking:
    """Sampled tracking and its exact positions on the original frame axis."""

    sequence: TrackingSequence
    source_positions: np.ndarray
    original_runs: tuple[tuple[int, int], ...]
    sampled_runs: tuple[tuple[int, int], ...]
    source_n_frames: int
    source_fps: float

    def __post_init__(self) -> None:
        positions = np.array(self.source_positions, dtype=np.float64, copy=True)
        original = tuple((int(start), int(stop)) for start, stop in self.original_runs)
        sampled = tuple((int(start), int(stop)) for start, stop in self.sampled_runs)
        if positions.shape != (len(self.sequence.ball_xy),) or not np.isfinite(positions).all():
            raise ValueError("source positions must be a finite sampled-frame vector")
        if (not isinstance(self.source_n_frames, (int, np.integer))
                or self.source_n_frames < 0):
            raise ValueError("source_n_frames must be a nonnegative integer")
        if not np.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("source_fps must be finite and positive")
        if len(original) != len(sampled):
            raise ValueError("source and sampled run counts differ")
        previous_source, previous_sample = 0, 0
        for (start, stop), (first, last) in zip(original, sampled):
            if start < previous_source or stop <= start or stop > self.source_n_frames:
                raise ValueError("original runs must be ordered, disjoint, and in bounds")
            if first != previous_sample or last <= first or last > len(positions):
                raise ValueError("sampled runs must partition the sampled frame vector")
            run_positions = positions[first:last]
            if (run_positions[0] != start or run_positions[-1] > stop - 1
                    or np.any(np.diff(run_positions) <= 0)):
                raise ValueError("sample positions must increase within their anchored run")
            previous_source, previous_sample = stop, last
        if previous_sample != len(positions):
            raise ValueError("sampled runs do not cover all sampled frames")
        if valid_runs(self.sequence.period) != sampled:
            raise ValueError("sampled periods must preserve every distinct run")
        positions.setflags(write=False)
        object.__setattr__(self, "source_positions", positions)
        object.__setattr__(self, "original_runs", original)
        object.__setattr__(self, "sampled_runs", sampled)
        object.__setattr__(self, "source_n_frames", int(self.source_n_frames))
        object.__setattr__(self, "source_fps", float(self.source_fps))


def _sample_positions(start: int, stop: int, source_fps: float,
                      target_fps: float) -> np.ndarray:
    step = source_fps / target_fps
    steps = (stop - start - 1) / step
    count = int(np.floor(np.nextafter(steps, np.inf))) + 1
    positions = start + np.arange(count, dtype=np.float64) * step
    # Floating-point arithmetic must not turn an exact endpoint into a blend.
    nearest = np.rint(positions)
    tolerance = 8 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(positions))
    exact = np.abs(positions - nearest) <= tolerance
    positions[exact] = nearest[exact]
    return positions[positions <= stop - 1]


def _interpolate_coordinates(values: np.ndarray, left: np.ndarray,
                             right: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = source[left].copy()
    blend = left != right
    if np.any(blend):
        first, last = source[left[blend]], source[right[blend]]
        weight_shape = (int(blend.sum()),) + (1,) * (source.ndim - 1)
        weight = fraction[blend].reshape(weight_shape)
        finite = np.isfinite(first).all(axis=-1) & np.isfinite(last).all(axis=-1)
        interpolated = first + weight * (last - first)
        output[blend] = np.where(finite[..., None], interpolated, np.nan)
    return output


def resample_tracking(sequence: TrackingSequence,
                      target_fps: float = 10.0) -> ResampledTracking:
    """Sample each valid run from its original start at ``target_fps``.

    Detected flags require both interpolation endpoints, or only the exact
    selected frame for an integer position.  Per-frame team IDs must agree
    across an interpolation pair; an ambiguous pair raises instead of inventing
    a player identity.  Exact native-grid sampling preserves those IDs verbatim.
    """
    if not isinstance(sequence, TrackingSequence):
        raise TypeError("sequence must be a TrackingSequence")
    rate = float(target_fps)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("target_fps must be finite and positive")
    source_rate = float(sequence.fps)
    original_runs = valid_runs(sequence.period)
    position_parts, period_parts, sampled_runs = [], [], []
    offset = 0
    for number, (start, stop) in enumerate(original_runs, start=1):
        positions = _sample_positions(start, stop, source_rate, rate)
        position_parts.append(positions)
        period_parts.append(np.full(len(positions), number, dtype=np.int32))
        sampled_runs.append((offset, offset + len(positions)))
        offset += len(positions)
    positions = (np.concatenate(position_parts) if position_parts
                 else np.empty(0, dtype=np.float64))
    periods = (np.concatenate(period_parts) if period_parts
               else np.empty(0, dtype=np.int32))
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    fraction = positions - left
    teams = np.asarray(sequence.player_team)
    if teams.ndim == 1:
        sampled_teams = teams.copy()
    else:
        if np.any(teams[left] != teams[right]):
            raise ValueError("cannot interpolate between different player team identities")
        sampled_teams = teams[left].copy()
    ball_detected = np.asarray(sequence.ball_detected, dtype=bool)
    player_detected = np.asarray(sequence.player_detected, dtype=bool)
    sampled_sequence = TrackingSequence(
        ball_xy=_interpolate_coordinates(sequence.ball_xy, left, right, fraction),
        ball_detected=ball_detected[left] & ball_detected[right],
        player_xy=_interpolate_coordinates(sequence.player_xy, left, right, fraction),
        player_detected=player_detected[left] & player_detected[right],
        player_team=sampled_teams,
        period=periods,
        fps=rate,
        pitch_m=tuple(sequence.pitch_m),
    )
    return ResampledTracking(
        sequence=sampled_sequence,
        source_positions=positions,
        original_runs=original_runs,
        sampled_runs=tuple(sampled_runs),
        source_n_frames=len(sequence.ball_xy),
        source_fps=source_rate,
    )


def project_probabilities(probabilities, mapping: ResampledTracking) -> np.ndarray:
    """Project probabilities to native frames, holding endpoints within each run.

    Frames outside original runs remain neutral (0.5).  No interpolation or
    endpoint extension crosses a run boundary.
    """
    if not isinstance(mapping, ResampledTracking):
        raise TypeError("mapping must be ResampledTracking")
    probability = np.asarray(probabilities, dtype=np.float64)
    if (probability.shape != mapping.source_positions.shape
            or not np.isfinite(probability).all()
            or np.any((probability < 0.0) | (probability > 1.0))):
        raise ValueError("probabilities must be a finite sampled-frame vector in [0, 1]")
    output = np.full(mapping.source_n_frames, 0.5, dtype=np.float64)
    for (start, stop), (first, last) in zip(mapping.original_runs, mapping.sampled_runs):
        output[start:stop] = np.interp(
            np.arange(start, stop, dtype=np.float64),
            mapping.source_positions[first:last],
            probability[first:last],
        )
    return output


__all__ = ["ResampledTracking", "resample_tracking", "project_probabilities"]
