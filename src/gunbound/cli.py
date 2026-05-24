"""
cli.py — Command-line interface for the GunBound shot calculator.

Entry points:
  python main.py                 # interactive calculator loop
  python main.py --calibrate     # recalibrate all mobiles from training data
  python main.py --validate      # print per-shot errors for all training shots
  python main.py --training      # record new shots into training data
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from .calibration import calibrate, recalibrate_all, validate
from .constants import KNOWN_MOBILES, POWER_MIN, POWER_MAX
from .matching import suggest_shots
from .physics import default_v_scale
from .storage import (
    PROJECT_ROOT,
    TRAINING_FILE,
    WIND_FILE,
    load_mobiles,
    load_training,
    save_mobiles,
    save_training,
)

try:
    from .position_capture import CaptureState, start_listener
    _HAS_POSITION_CAPTURE = True
except ImportError:
    _HAS_POSITION_CAPTURE = False
    CaptureState = None  # type: ignore[assignment,misc]
    start_listener = lambda s: None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Terminal color support
# ─────────────────────────────────────────────────────────────────────────────

def _init_colors():
    """Return an ANSI color namespace; falls back to empty strings on non-TTY."""
    os.system("")  # Activate VT100 processing on Windows 10+
    if not sys.stdout.isatty():
        class _C:
            RESET = CYAN = YELLOW = GREEN = WHITE = DIM = ""
        return _C()
    class _C:
        RESET  = "\033[0m"
        CYAN   = "\033[96m"
        YELLOW = "\033[93m"
        GREEN  = "\033[92m"
        WHITE  = "\033[97m"
        DIM    = "\033[2m"
    return _C()

_Color = _init_colors()


# ─────────────────────────────────────────────────────────────────────────────
# Wind reader helpers
# ─────────────────────────────────────────────────────────────────────────────

def _start_wind_reader(reader: str = "classic") -> subprocess.Popen | None:
    """Spawn wind reader as a background process. Returns Popen or None.

    reader: "classic" → tools/wind_reader.py  (fixed-position, original)
            "gitzwc"  → tools/wind_reader_gitzwc.py  (dynamic auto-detect)
    """
    script_name = "wind_reader_gitzwc.py" if reader == "gitzwc" else "wind_reader.py"
    script = PROJECT_ROOT / "tools" / script_name
    if not script.exists():
        return None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except OSError:
        return None


def _prompt_wind_reader_choice() -> str:
    """Ask the player which wind reader to use. Returns 'classic' or 'gitzwc'.

    Falls back to 'classic' silently if stdin is not a TTY.
    """
    if not sys.stdin.isatty():
        return "classic"
    print("  Wind reader:  [1] Classic (fixed position)  [2] GitzWC (auto-detect)")
    try:
        choice = input("  Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "classic"
    return "gitzwc" if choice == "2" else "classic"


def _read_wind_angle() -> float | None:
    """Read wind angle from data/wind.json. Returns None if unavailable."""
    try:
        data = json.loads(Path(WIND_FILE).read_text(encoding="utf-8"))
        return float(data["angle"])
    except (OSError, KeyError, ValueError):
        return None


def _read_wind_strength() -> int | None:
    """Read wind strength from data/wind.json. Returns None if unavailable or negative."""
    try:
        data = json.loads(Path(WIND_FILE).read_text(encoding="utf-8"))
        s = int(data["strength"])
        return s if s >= 0 else None
    except (OSError, KeyError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_captured_block(
    direction_label: str,
    target_slices: float,
    target_sd: float,
    height_slices: float,
    height_diff: float,
    wind_angle: float,
    wind_strength: int | float,
) -> None:
    """Print the multi-line Captured: summary block with optional color."""
    c = _Color
    print(f"{c.YELLOW}Captured:{c.RESET}")
    print(f"{c.YELLOW}  direction={direction_label}{c.RESET}")
    print(f"{c.YELLOW}  target={target_slices:.1f} slices ({target_sd:.3f} SD){c.RESET}")
    print(f"{c.YELLOW}  height={height_slices:+.1f} slices ({height_diff:+.3f} SD){c.RESET}")
    print(f"{c.YELLOW}  Wind angle: {wind_angle:+.1f}°{c.RESET}")
    print(f"{c.YELLOW}  Wind strength: {wind_strength}{c.RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        try:
            val = float(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  Enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Invalid — enter a number.")


def _prompt_choice(prompt: str, choices: list[str]) -> str:
    choices_lower = [c.lower() for c in choices]
    while True:
        val = input(prompt).strip().lower()
        if val in choices_lower:
            return val
        print(f"  Choose from: {', '.join(choices)}")


# ─────────────────────────────────────────────────────────────────────────────
# Training mode
# ─────────────────────────────────────────────────────────────────────────────

def training_mode(mobiles_cfg: dict) -> None:
    """CLI loop: manually record known shots into data/training_data.json."""
    training_data = load_training()

    print("=== GunBound Calculator — Training mode ===")
    print(f"  Known mobiles: {', '.join(KNOWN_MOBILES)}")

    while True:
        mobile = input("Mobile: ").strip().lower()
        if mobile in KNOWN_MOBILES:
            break
        print(f"  Unknown mobile. Choose from: {', '.join(KNOWN_MOBILES)}")

    added = 0
    while True:
        try:
            print()
            angle         = _prompt_float("Angle (degrees): ", 0.0, 90.0)
            power         = _prompt_float("Power (bars): ", POWER_MIN, POWER_MAX)
            wind_strength = _prompt_float("Wind strength (0–26): ", 0.0, 26.0)
            wind_angle    = _prompt_float(
                "Wind angle (0=up, 90=toward enemy, -90=away, ±180=down): ", -180.0, 180.0
            )
            height_slices = _prompt_float(
                "Height diff (slices, positive = you are higher): ", -8.0, 8.0
            )
            height_diff   = height_slices * 0.125
            actual_slices = _prompt_float("Actual landing (slices): ", 0.4, 40.0)
            actual_sd     = actual_slices * 0.125

            sample = {
                "mobile":        mobile,
                "angle":         angle,
                "power":         power,
                "wind_strength": wind_strength,
                "wind_angle":    wind_angle,
                "height_diff":   height_diff,
                "actual_sd":     actual_sd,
            }
            training_data.append(sample)
            save_training(training_data)
            added += 1
            print(f"  Recorded. ({added} this session, {len(training_data)} total)")

            again = input("  Another shot? (y/n): ").strip().lower()
            if again != "y":
                break
        except (ValueError, EOFError, KeyboardInterrupt):
            break

    if added > 0:
        shots_for_mobile = [d for d in training_data if d["mobile"] == mobile]
        print(f"\n  Recalibrating {mobile} on {len(shots_for_mobile)} shot(s)…")
        v_scale, power_exp, wind_x, wind_y = calibrate(shots_for_mobile, mobile)
        if mobile not in mobiles_cfg:
            mobiles_cfg[mobile] = {}
        mobiles_cfg[mobile]["v_scale"]      = round(v_scale,    6)
        mobiles_cfg[mobile]["power_exp"]    = round(power_exp,  6)
        mobiles_cfg[mobile]["wind_x_coeff"] = round(wind_x,     6)
        mobiles_cfg[mobile]["wind_y_coeff"] = round(wind_y,     6)
        save_mobiles(mobiles_cfg)
        print(f"  v_scale={v_scale:.4f}  power_exp={power_exp:.4f}  wind_x={wind_x:.4f}  wind_y={wind_y:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main calculator loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── CLI flags ──────────────────────────────────────────────────────────
    known_flags = {"--calibrate", "--validate", "--training", "--infer-priors", "--dry-run", "--wind-reader"}
    # Use a.split("=")[0] so --wind-reader=gitzwc is accepted by the validator.
    unknown = [a for a in sys.argv[1:] if a.startswith("--") and a.split("=")[0] not in known_flags]
    if unknown:
        print(f"Unknown flag(s): {', '.join(unknown)}")
        print(f"  Valid flags: {', '.join(sorted(known_flags))}")
        sys.exit(1)

    # Parse optional --wind-reader {classic|gitzwc} (both --flag value and --flag=value forms).
    _wr: str | None = None
    for _i, _a in enumerate(sys.argv):
        if _a.startswith("--wind-reader="):
            _wr = _a.split("=", 1)[1]
            break
        if _a == "--wind-reader" and _i + 1 < len(sys.argv):
            _wr = sys.argv[_i + 1]
            break
    wind_reader_arg: str | None = _wr if _wr in ("classic", "gitzwc") else None

    if "--infer-priors" in sys.argv:
        dry_run = "--dry-run" in sys.argv
        print("\n=== GunBound Calculator — Infer Priors ===")
        cfg = load_mobiles()
        armor = cfg.get("armor", {})
        if armor.get("power_exp") is None:
            print("  ERROR: Armor is not fully calibrated — run --calibrate first.")
            sys.exit(1)
        from .inference import compute_priors, apply_priors
        priors = compute_priors(cfg)
        print(
            f"  Armor reference: "
            f"v_scale={armor['v_scale']:.4f}  "
            f"wind_x={armor['wind_x_coeff']:.4f}  "
            f"wind_y={armor['wind_y_coeff']:.4f}  "
            f"power_exp={armor['power_exp']:.4f}"
        )
        print()
        header = f"  {'mobile':<12}  {'v_scale':>7}  {'power_exp':>9}  {'wind_x':>6}  {'wind_y':>6}  status"
        print(header)
        print("  " + "─" * (len(header) - 2))
        updated_count = 0
        for mobile, prior in sorted(priors.items()):
            existing = cfg.get(mobile, {})
            fitted = "power_exp" in existing
            status = "[SKIPPED — fully calibrated]" if fitted else "[UPDATED]"
            if not fitted:
                updated_count += 1
            print(
                f"  {mobile:<12}  {prior['v_scale']:>7.4f}  "
                f"{prior['power_exp']:>9.4f}  "
                f"{prior['wind_x_coeff']:>6.4f}  "
                f"{prior['wind_y_coeff']:>6.4f}  {status}"
            )
        print()
        if dry_run:
            print(f"  [DRY-RUN] No files were modified.")
        else:
            apply_priors(cfg)
            print(f"  {updated_count} mobile(s) updated. Saved to config/mobiles_v2.json.")
        return

    if "--calibrate" in sys.argv:
        print("\n=== GunBound Calculator — Calibration ===")
        cfg           = load_mobiles()
        training_data = load_training()
        print(f"  Loaded {len(training_data)} training shot(s) from {TRAINING_FILE}")
        recalibrate_all(cfg, training_data)
        print("\nDone.")
        return

    if "--validate" in sys.argv:
        print("\n=== GunBound Calculator — Validation ===")
        cfg           = load_mobiles()
        training_data = load_training()
        print(f"  Loaded {len(training_data)} shot(s) | {len(cfg)} calibrated mobiles\n")
        validate(cfg, training_data)
        return

    mobiles_cfg   = load_mobiles()
    training_data = load_training()

    if "--training" in sys.argv:
        training_mode(mobiles_cfg)
        return

    # ── Normal calculator loop ─────────────────────────────────────────────
    print("\n=== GunBound Shot Calculator ===")
    print(f"  Calibrated mobiles: {', '.join(sorted(mobiles_cfg.keys()))}")
    print(f"  Training shots loaded: {len(training_data)}")
    print("  Type 'exit' to quit.\n")

    # Mobile — asked once at startup
    while True:
        mobile_input = input("Mobile: ").strip().lower()
        if mobile_input in ("exit", "quit", "q"):
            return
        if mobile_input in mobiles_cfg:
            break
        if mobile_input in KNOWN_MOBILES:
            print(f"  '{mobile_input}' not calibrated yet — using derived defaults.")
            armor_vs = mobiles_cfg.get("armor", {}).get("v_scale", 1.45)
            mobiles_cfg[mobile_input] = {
                "v_scale":      default_v_scale(mobile_input, armor_vs),
                "wind_x_coeff": 0.10,
                "wind_y_coeff": 0.10,
            }
            break
        print(f"  Unknown mobile '{mobile_input}'. Known: {', '.join(KNOWN_MOBILES)}")

    print(f"  Mobile set to: {mobile_input.upper()}\n")

    # Start wind reader in background
    reader_choice = wind_reader_arg or _prompt_wind_reader_choice()
    wind_proc = _start_wind_reader(reader=reader_choice)
    if wind_proc:
        print(f"  Wind reader started ({reader_choice}) in background.\n")

    # Start position capture hotkey listener
    from .constants import HOTKEY_OWN_LABEL, HOTKEY_TARGET_LABEL
    capture_state = CaptureState() if _HAS_POSITION_CAPTURE else None
    if capture_state is not None:
        start_listener(capture_state)
        print(f"  Hotkeys active: {HOTKEY_OWN_LABEL} = mark own position, {HOTKEY_TARGET_LABEL} = mark target.\n")

    while True:
        try:
            # ── Position capture override (hotkey pre-filled) ──────────────────
            pair = capture_state.consume() if capture_state is not None else None
            if pair is not None and not pair.is_valid:
                print("  [Capture] Positions out of range — using manual input.")
                pair = None

            if pair is not None:
                # Both positions captured and valid — skip direction/SD/height prompts
                target_slices = pair.target_slices
                target_sd   = target_slices * 0.125
                height_slices = pair.height_slices
                height_diff = height_slices * 0.125
                shoot_right = pair.shoot_right
                direction_label = "RIGHT" if shoot_right else "LEFT"
            else:
                # Manual input — also re-checks capture on each Enter (handles the
                # case where hotkeys were pressed while this prompt was blocking)
                while True:
                    d = input("Shooting direction (L/R): ").strip().lower()

                    # Re-check: user may have pressed hotkeys while input() was blocking
                    re_pair = capture_state.consume() if capture_state is not None else None
                    if re_pair is not None and not re_pair.is_valid:
                        print("  [Capture] Positions out of range — using manual input.")
                        re_pair = None
                    if re_pair is not None:
                        # Switch to capture path mid-prompt
                        pair = re_pair
                        break

                    if d in ("l", "r", "left", "right"):
                        shoot_right = d.startswith("r")
                        break
                    if d:
                        print("  Enter L (left) or R (right).")
                    # empty input (user pressed Enter after hotkeys): loop silently

                if pair is not None:
                    # Capture path (same as the outer capture block above)
                    target_slices = pair.target_slices
                    target_sd   = target_slices * 0.125
                    height_slices = pair.height_slices
                    height_diff = height_slices * 0.125
                    shoot_right = pair.shoot_right
                    direction_label = "RIGHT" if shoot_right else "LEFT"
                else:
                    target_slices = _prompt_float("Target (slices, 1 slice = 0.125 SD): ", 0.8, 24.0)
                    target_sd     = target_slices * 0.125
                    height_slices = _prompt_float(
                        "Height diff (slices, positive = you are higher, -1.0 to 1.0): ", -8.0, 8.0
                    )
                    height_diff   = height_slices * 0.125
                    direction_label = "RIGHT" if shoot_right else "LEFT"
            wind_angle = _read_wind_angle()
            if wind_angle is not None:
                # wind_reader: 0=East, CCW, 0–360
                # GunBound:    0=up, CW, ±180 (positive = toward enemy)
                # Assuming enemy is to the right: gb = 90 - reader (normalised to ±180)
                gb = (90.0 - wind_angle + 180.0) % 360.0 - 180.0
                if not shoot_right:
                    gb = -gb
                wind_angle = gb
            else:
                wind_angle = _prompt_float(
                    "Wind angle (0=up, 90=toward, -90=away, ±180=down): ", -180.0, 180.0
                )
            wind_strength = int(_prompt_float("Wind strength: ", 0.0, 26.0))
            _print_captured_block(
                direction_label,
                target_slices,
                target_sd,
                height_slices,
                height_diff,
                wind_angle,
                wind_strength,
            )
        except (EOFError, KeyboardInterrupt):
            break

        shots = suggest_shots(
            mobiles_cfg, mobile_input, target_sd,
            wind_strength, wind_angle, height_diff,
            training_data=training_data,
        )

        if not shots:
            print("  No solution found — check inputs.\n")
            continue

        print(f"\n  Suggestions @ {target_sd:.3f} SD  "
              f"(wind {wind_strength}@{wind_angle:+.0f}° height {height_diff:+.2f}):")
        for i, r in enumerate(shots, 1):
            if r.source == "data":
                tag   = f"data: {r.n_samples} shot(s), ±{r.error:.3f} SD spread"
                color = _Color.GREEN
            else:
                tag   = f"physics, err {r.error:+.4f} SD"
                color = _Color.WHITE
            print(f"{color}    {i}) {r.angle:.0f}°  power={r.power:.2f}  [{tag}]{_Color.RESET}")
        print()

        # Optional: record the shot used
        choice = input("  Which did you use? (Enter to skip): ").strip()
        if choice in [str(i) for i in range(1, len(shots) + 1)]:
            chosen = shots[int(choice) - 1]
            try:
                actual_slices = _prompt_float("  Where did it land (slices)? ", 0.4, 40.0)
                actual_sd = actual_slices * 0.125
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            sample = {
                "mobile":        mobile_input,
                "angle":         chosen.angle,
                "power":         chosen.power,
                "wind_strength": wind_strength,
                "wind_angle":    wind_angle,
                "height_diff":   height_diff,
                "actual_sd":     actual_sd,
            }
            training_data.append(sample)
            save_training(training_data)
            print(f"  Recorded. ({len(training_data)} total shots)")

            # Auto-recalibrate after every 5 new shots
            mobile_shots = [d for d in training_data if d["mobile"] == mobile_input]
            if len(mobile_shots) % 5 == 0 and len(mobile_shots) >= 3:
                print(f"  Auto-recalibrating {mobile_input} ({len(mobile_shots)} shots)…", end=" ")
                vs, pe, wx, wy = calibrate(mobile_shots, mobile_input)
                mobiles_cfg[mobile_input]["v_scale"]      = round(vs, 6)
                mobiles_cfg[mobile_input]["power_exp"]    = round(pe, 6)
                mobiles_cfg[mobile_input]["wind_x_coeff"] = round(wx, 6)
                mobiles_cfg[mobile_input]["wind_y_coeff"] = round(wy, 6)
                save_mobiles(mobiles_cfg)
                print(f"v_scale={vs:.4f}  power_exp={pe:.4f}  wind_x={wx:.4f}  wind_y={wy:.4f}")
        print()


if __name__ == "__main__":
    main()
