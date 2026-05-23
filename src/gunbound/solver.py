"""
solver.py — Physics-based angle/power solver for the GunBound shot calculator.

The solver uses a coarse → refine strategy:
  1. Coarse pass: sweeps every SOLVER_COARSE_STEP degrees.
  2. Refine pass: sweeps every integer degree within ±3° of each coarse hit.
  3. De-duplicate adjacent angles with nearly identical power.
  4. Prefer the 45–80° range as the most practical GunBound window.
"""

from .constants import (
    ANGLE_MIN, ANGLE_MAX, POWER_MIN, POWER_MAX,
    MAX_SUGGESTIONS, SOLVER_COARSE_STEP, SOLVER_POWER_STEPS,
    MOBILE_ANGLE_RANGE, CLOSE_RANGE_THRESHOLD, HIGH_ANGLE_MIN,
)
from .models import ShotResult
from .physics import simulate_shot


def _solve_for_target(
    target_sd: float,
    mobile: str,
    cfg: dict,
    wind_strength: float,
    wind_angle_deg: float,
    height_diff: float,
    angle_step: float,
    power_steps: int,
    angle_subset: set[int] | None = None,
    angle_min: int = ANGLE_MIN,
    angle_max: int = ANGLE_MAX,
) -> list[ShotResult]:
    """Sweep angles and binary-search power for each angle."""
    v_scale   = cfg.get("v_scale",      1.45)
    power_exp = cfg.get("power_exp",    1.0)
    wind_x    = cfg.get("wind_x_coeff", 0.10)
    wind_y    = cfg.get("wind_y_coeff", 0.10)

    results = []

    def _sim(a, p):
        return simulate_shot(a, p, mobile, v_scale, power_exp, wind_x, wind_y,
                             wind_strength, wind_angle_deg, height_diff)

    if angle_subset is not None:
        angles = sorted(angle_subset)
    else:
        step = max(1, int(angle_step))
        angles = list(range(angle_min, angle_max + 1, step))

    for angle in angles:
        lo, hi = POWER_MIN, POWER_MAX
        x_lo   = _sim(angle, lo)
        x_hi   = _sim(angle, hi)

        # Skip if target is outside the reachable range for this angle
        if target_sd < min(x_lo, x_hi) - 0.05:
            continue
        if target_sd > max(x_lo, x_hi) + 0.05:
            continue

        # Binary search for power
        for _ in range(30):
            mid   = (lo + hi) / 2
            x_mid = _sim(angle, mid)
            if x_mid < target_sd:
                lo = mid
            else:
                hi = mid

        best_p = (lo + hi) / 2
        err    = _sim(angle, best_p) - target_sd
        results.append(ShotResult(angle=float(angle), power=round(best_p, 2), error=err))

    return results


def solve(
    target_sd: float,
    mobile: str,
    cfg: dict,
    wind_strength: float,
    wind_angle_deg: float,
    height_diff: float,
) -> list[ShotResult]:
    """Coarse → refine solver.

    Returns up to MAX_SUGGESTIONS ShotResult objects sorted by angle.
    """
    angle_min, angle_max = MOBILE_ANGLE_RANGE.get(mobile, (ANGLE_MIN, ANGLE_MAX))

    # Coarse pass
    coarse = _solve_for_target(
        target_sd, mobile, cfg, wind_strength, wind_angle_deg, height_diff,
        angle_step=SOLVER_COARSE_STEP, power_steps=SOLVER_POWER_STEPS,
        angle_min=angle_min, angle_max=angle_max,
    )
    # Restrict coarse results to the mobile's allowed angle range
    coarse = [r for r in coarse if angle_min <= r.angle <= angle_max]

    # Refine: expand ±3° around each coarse hit, sweep every integer degree
    refined_angles: set[int] = set()
    for r in coarse:
        for da in range(-3, 4):
            a = int(r.angle) + da
            if angle_min <= a <= angle_max:
                refined_angles.add(a)

    # If no coarse hits at all, sweep everything at integer steps
    if not refined_angles:
        refined_angles = set(range(angle_min, angle_max + 1))

    refined = _solve_for_target(
        target_sd, mobile, cfg, wind_strength, wind_angle_deg, height_diff,
        angle_step=1, power_steps=SOLVER_POWER_STEPS * 2,
        angle_subset=refined_angles,
        angle_min=angle_min, angle_max=angle_max,
    )

    # Keep only the best power per integer angle
    best_by_angle: dict[int, ShotResult] = {}
    for r in refined:
        key = int(round(r.angle))
        if key not in best_by_angle or abs(r.error) < abs(best_by_angle[key].error):
            best_by_angle[key] = r

    results = sorted(best_by_angle.values(), key=lambda r: r.angle)

    # De-duplicate: suppress adjacent angles with nearly identical power
    # Two shots suppressed if within ±3° AND power difference < 0.08
    deduped = []
    for r in results:
        if deduped and abs(r.angle - deduped[-1].angle) <= 3 and abs(r.power - deduped[-1].power) < 0.08:
            continue
        deduped.append(r)

    # Prefer a spread across the angle range; pick up to MAX_SUGGESTIONS
    # biased toward the "useful" GunBound range (roughly 45–80°).
    # When shooter is below target (height_diff < 0), low angles are impractical:
    # raise the minimum preferred angle proportionally to the height disadvantage.
    pref_min = 45
    if height_diff < -0.1:
        pref_min = 70
    elif height_diff < -0.05:
        pref_min = 60
    elif height_diff < -0.02:
        pref_min = 55

    preferred = [r for r in deduped if pref_min <= r.angle <= 80]
    others    = [r for r in deduped if r.angle < pref_min or r.angle > 80]

    # Pick evenly-spaced suggestions from preferred range
    if len(preferred) > MAX_SUGGESTIONS:
        step = len(preferred) / MAX_SUGGESTIONS
        preferred = [preferred[int(i * step)] for i in range(MAX_SUGGESTIONS)]

    chosen = preferred[:MAX_SUGGESTIONS]
    remaining_slots = MAX_SUGGESTIONS - len(chosen)
    if remaining_slots > 0:
        chosen = others[:remaining_slots] + chosen

    # For close-range shots, always include the highest viable angle (≥ HIGH_ANGLE_MIN)
    # so the player has a steep-arc option. Replaces the last chosen slot if needed.
    if target_sd < CLOSE_RANGE_THRESHOLD:
        high_angle_candidates = [
            r for r in deduped
            if r.angle >= HIGH_ANGLE_MIN and r not in chosen
        ]
        if high_angle_candidates:
            best_high = max(high_angle_candidates, key=lambda r: r.angle)
            if len(chosen) >= MAX_SUGGESTIONS:
                chosen[-1] = best_high   # replace last (lowest-priority) slot
            else:
                chosen.append(best_high)

    return sorted(chosen, key=lambda r: r.angle)
