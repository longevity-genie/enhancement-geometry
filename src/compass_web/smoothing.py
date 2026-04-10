"""Radii and spacing smoothing to prevent steep loft slopes.

When consecutive radii differ by more than a 2.5:1 ratio (larger >= 2.5x smaller),
the loft surface has a near-vertical slope that produces degenerate voronoi
cell intersections.  This module detects such transitions and corrects them:

- The smaller radius is increased so the ratio stays at most 2:1
- Spacing is adjusted for 25° / 50° profile rules (see README)
- If total Z span exceeds ``HEIGHT_LIMIT_FACTOR`` times the original
  ``(n-1) * z_increment``, radii are nudged in small steps toward the original
  input and the ratio+geometry passes are re-run until within limit or max
  iterations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from compass_web.config import MAX_RADIUS, MIN_RADIUS

MAX_CONSECUTIVE_RATIO = 2.5
TARGET_RATIO = 2.0
MIN_ANGLE_FROM_VERTICAL_DEG = 25.0
MIN_ANGLE_BETWEEN_SEGMENTS_DEG = 50.0
MIN_RELATIVE_DR = 1e-4
VERTEX_WIDEN_FACTOR = 1.03
MAX_GEOMETRY_PASSES = 3000
HEIGHT_LIMIT_FACTOR = 2.0
RADII_NUDGE_TOWARD_ORIGINAL = 0.02
MAX_HEIGHT_NUDGE_ROUNDS = 120


def _tan_min_vertical_angle() -> float:
    return math.tan(math.radians(MIN_ANGLE_FROM_VERTICAL_DEG))


def _vertical_angle_spacing_cap(r_a: float, r_b: float) -> float | None:
    dr = abs(r_b - r_a)
    if dr <= 0.0:
        return None
    scale = max(abs(r_a), abs(r_b), 1e-12)
    if dr / scale < MIN_RELATIVE_DR:
        return None
    return dr / _tan_min_vertical_angle()


def _max_dz_for_segment(radii: list[float], i: int) -> float:
    cap = _vertical_angle_spacing_cap(radii[i], radii[i + 1])
    return cap if cap is not None else float("inf")


def _segment_turn_angle_deg(
    r_prev: float,
    r_mid: float,
    r_next: float,
    dz_prev: float,
    dz_next: float,
) -> float:
    v1 = (r_mid - r_prev, dz_prev)
    v2 = (r_next - r_mid, dz_next)
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 <= 0.0 or n2 <= 0.0:
        return 180.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    c = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(c))


def _ratio_pass(
    adjusted: list[float],
    *,
    max_ratio: float,
    target_ratio: float,
    adjustments: list[str],
) -> list[bool]:
    n = len(adjusted)
    steep_interval = [False] * (n - 1)
    for i in range(n - 1):
        r_a = adjusted[i]
        r_b = adjusted[i + 1]
        larger = max(r_a, r_b)
        smaller = min(r_a, r_b)
        if smaller <= 0:
            continue
        ratio = larger / smaller
        if ratio < max_ratio:
            continue
        new_smaller = larger / target_ratio
        steep_interval[i] = True
        if r_a < r_b:
            old_val = adjusted[i]
            adjusted[i] = new_smaller
            adjustments.append(
                f"R{i + 1}: {old_val:.3f} -> {new_smaller:.3f} "
                f"(ratio with R{i + 2} was {ratio:.1f}x, now {target_ratio:.1f}x)"
            )
        else:
            old_val = adjusted[i + 1]
            adjusted[i + 1] = new_smaller
            adjustments.append(
                f"R{i + 2}: {old_val:.3f} -> {new_smaller:.3f} "
                f"(ratio with R{i + 1} was {ratio:.1f}x, now {target_ratio:.1f}x)"
            )
    return steep_interval


def _geometry_pass(
    adjusted: list[float],
    spacings: list[float],
    steep_interval: list[bool],
    adjustments: list[str],
) -> int:
    n = len(adjusted)
    vertex_widen_steps = 0
    if not any(steep_interval):
        return vertex_widen_steps
    for _ in range(MAX_GEOMETRY_PASSES):
        changed = False
        for i in range(n - 1):
            cap = _vertical_angle_spacing_cap(adjusted[i], adjusted[i + 1])
            if cap is None:
                continue
            if spacings[i] > cap + 1e-9:
                old_sp = spacings[i]
                spacings[i] = cap
                changed = True
                tag = "steep interval" if steep_interval[i] else "adjacent interval"
                adjustments.append(
                    f"Spacing {i + 1}-{i + 2}: {old_sp:.3f} -> {cap:.3f} "
                    f"(>={MIN_ANGLE_FROM_VERTICAL_DEG:.0f} deg from vertical, {tag})"
                )
        if n < 3:
            if not changed:
                break
            continue
        worst_j = -1
        worst_ang = 180.0
        for j in range(1, n - 1):
            if not (steep_interval[j - 1] or steep_interval[j]):
                continue
            ang = _segment_turn_angle_deg(
                adjusted[j - 1],
                adjusted[j],
                adjusted[j + 1],
                spacings[j - 1],
                spacings[j],
            )
            if ang < MIN_ANGLE_BETWEEN_SEGMENTS_DEG - 1e-3 and ang < worst_ang:
                worst_ang = ang
                worst_j = j
        if worst_j < 0:
            if not changed:
                break
            continue
        j = worst_j
        max_prev = _max_dz_for_segment(adjusted, j - 1)
        max_next = _max_dz_for_segment(adjusted, j)
        new_prev = min(spacings[j - 1] * VERTEX_WIDEN_FACTOR, max_prev)
        new_next = min(spacings[j] * VERTEX_WIDEN_FACTOR, max_next)
        new_prev = max(new_prev, spacings[j - 1])
        new_next = max(new_next, spacings[j])
        if new_prev <= spacings[j - 1] + 1e-9 and new_next <= spacings[j] + 1e-9:
            if not changed:
                break
            continue
        spacings[j - 1] = new_prev
        spacings[j] = new_next
        vertex_widen_steps += 1
        changed = True
    return vertex_widen_steps


def _clamp_radii(radii: list[float]) -> None:
    for i, r in enumerate(radii):
        radii[i] = max(MIN_RADIUS, min(MAX_RADIUS, float(r)))


def _run_ratio_and_geometry(
    adjusted: list[float],
    z_increment: float,
    *,
    max_ratio: float,
    target_ratio: float,
    adjustments: list[str],
) -> list[float]:
    n = len(adjusted)
    spacings = [z_increment] * (n - 1)
    steep_interval = _ratio_pass(
        adjusted, max_ratio=max_ratio, target_ratio=target_ratio, adjustments=adjustments
    )
    vw = 0
    if any(steep_interval):
        vw = _geometry_pass(adjusted, spacings, steep_interval, adjustments)
        if vw > 0:
            adjustments.append(
                f"Profile vertices: {vw} widen step(s) "
                f"(x{VERTEX_WIDEN_FACTOR} per step, max dz per segment; "
                f"target >={MIN_ANGLE_BETWEEN_SEGMENTS_DEG:.0f} deg between segments)"
            )
    return spacings


@dataclass(frozen=True)
class SmoothingResult:
    """Result of radii/spacing smoothing."""

    original_radii: tuple[float, ...]
    original_z_increment: float
    adjusted_radii: tuple[float, ...]
    adjusted_z_levels: tuple[float, ...]
    adjusted_z_increment: float
    adjustments: tuple[str, ...]
    was_adjusted: bool


def smooth_radii_and_spacing(
    radii: tuple[float, ...],
    z_increment: float,
    *,
    max_ratio: float = MAX_CONSECUTIVE_RATIO,
    target_ratio: float = TARGET_RATIO,
) -> SmoothingResult:
    n = len(radii)
    original_input = tuple(radii)
    orig_height = (n - 1) * z_increment
    adjusted = list(radii)
    adjustments: list[str] = []

    spacings = _run_ratio_and_geometry(
        adjusted,
        z_increment,
        max_ratio=max_ratio,
        target_ratio=target_ratio,
        adjustments=adjustments,
    )

    height_limit_round = 0
    while (
        sum(spacings) > HEIGHT_LIMIT_FACTOR * orig_height + 1e-6
        and height_limit_round < MAX_HEIGHT_NUDGE_ROUNDS
    ):
        height_limit_round += 1
        adjustments.append(
            f"Height limit: total Z {sum(spacings):.3f} > {HEIGHT_LIMIT_FACTOR * orig_height:.3f}; "
            f"nudging radii toward original (round {height_limit_round})"
        )
        for i in range(n):
            adjusted[i] = (1.0 - RADII_NUDGE_TOWARD_ORIGINAL) * adjusted[i] + RADII_NUDGE_TOWARD_ORIGINAL * float(
                original_input[i]
            )
        _clamp_radii(adjusted)
        spacings = _run_ratio_and_geometry(
            adjusted,
            z_increment,
            max_ratio=max_ratio,
            target_ratio=target_ratio,
            adjustments=adjustments,
        )

    z_levels = [0.0]
    for sp in spacings:
        z_levels.append(z_levels[-1] + sp)

    was_adjusted = len(adjustments) > 0
    max_spacing = max(spacings) if spacings else z_increment

    return SmoothingResult(
        original_radii=original_input,
        original_z_increment=z_increment,
        adjusted_radii=tuple(adjusted),
        adjusted_z_levels=tuple(z_levels),
        adjusted_z_increment=max_spacing,
        adjustments=tuple(adjustments),
        was_adjusted=was_adjusted,
    )


def apply_smoothing_to_config(config: "PipelineConfig") -> tuple["PipelineConfig", SmoothingResult]:
    """Apply smoothing to a PipelineConfig, returning the adjusted config and the result."""
    from compass_web.config import PipelineConfig

    result = smooth_radii_and_spacing(config.radii, config.z_increment)
    if not result.was_adjusted:
        return config, result

    adjusted_config = replace(
        config,
        radii=result.adjusted_radii,
        z_increment=result.adjusted_z_increment,
        z_levels=result.adjusted_z_levels,
    )
    return adjusted_config, result
