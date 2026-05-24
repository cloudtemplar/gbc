"""
wind_reader_gitzwc.py — GitzWC dynamic-position wind rose reader.

The GitzWC private server repositions the wind rose widget to a new screen
location every match.  This script:

  1. Locates the wind rose anywhere on screen via cv2.matchTemplate against a
     bundled bezel-only template (assets/wind_rose_template.png + ring mask).
  2. Caches the detected bounding box and polls the angle using the same
     brightness × warmth radial-scoring algorithm from wind_reader.py.
  3. Writes data/wind.json in the same schema consumed by main.py.

Requirements:
    pip install opencv-python mss numpy
    (Pillow is a transitive dependency via wind_reader.py helpers)

Usage:
    python tools/wind_reader_gitzwc.py                   # poll loop
    python tools/wind_reader_gitzwc.py --capture-template  # capture template assets
    python tools/wind_reader_gitzwc.py --debug            # single-frame debug image
    Ctrl+C to stop.

Template capture workflow:
    1. Launch GitzWC in "no background" mode and start a match.
    2. Run with --capture-template.  An annotated full-screen window opens.
    3. Click the centre of the wind rose, then press any key to confirm.
    4. Both assets/wind_rose_template.png and assets/wind_rose_mask.png are saved.
"""

import sys
import time
import argparse

# Force UTF-8 output so Unicode characters don't crash on Windows cp1252 terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataclasses import dataclass
from pathlib import Path

# ── Dependency checks ────────────────────────────────────────────────────────

def _require(module_name: str, pip_name: str):
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"ERROR: '{pip_name}' is not installed.  Run: pip install {pip_name}")
        sys.exit(1)

cv2 = _require("cv2", "opencv-python")
np  = _require("numpy", "numpy")
mss_mod = _require("mss", "mss")
import mss as mss_pkg

# ── Shared helpers from wind_reader.py ─────────────────────────────────────
# Import via sys.path insert — wind_reader.py is not a package.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from wind_reader import (  # noqa: E402
    detect_direction_shape,
    WindBuffer,
    write_wind
)

# PIL is a transitive dep (used by detect_direction_shape).
from PIL import Image  # noqa: E402

# ── Path constants ─────────────────────────────────────────────────────────
# Define all paths locally — do NOT import from src/gunbound/storage.py so
# this tool remains independent of the core package install state.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT       = _SCRIPT_DIR.parent
ASSETS_DIR  = _ROOT / "assets"
DATA_DIR    = _ROOT / "data"

TEMPLATE_PATH = ASSETS_DIR / "wind_rose_template.png"
MASK_PATH     = ASSETS_DIR / "wind_rose_mask.png"
WIND_JSON     = DATA_DIR / "wind.json"
DEBUG_IMG     = DATA_DIR / "wind_debug_gitzwc.png"

# ── Tunable constants ──────────────────────────────────────────────────────

# Minimum TM_CCOEFF_NORMED score to trust a cached or newly scanned position.
# No mask is used for matching — the full 110×110 template provides enough
# features (bezel + background context).  Needle movement affects ~36% of pixels
# so the score at the correct location typically lands in the 0.55–0.85 range.
MATCH_THRESHOLD  = 0.50

# Per-poll re-check threshold.  Lower than MATCH_THRESHOLD because the needle
# may have rotated since the template was captured.
TRACK_THRESHOLD  = 0.35

# Consecutive frames below MATCH_THRESHOLD before a full-screen re-scan fires.
RESCAN_MISS_LIMIT = 10

# Configurable sub-pixel offset from the geometric template centre to the true
# disc centre.  Set to non-zero if your capture geometry is off-centre.
CENTER_OFFSET_X = 0
CENTER_OFFSET_Y = 0

# Seconds between poll cycles (matches wind_reader.py default).
POLL_INTERVAL = 0.3

# Rolling buffer depth for angle consensus (matches wind_reader.py default).
BUFFER_SIZE = 5

# Template capture: side length of the bounding-box crop (px).
CAPTURE_SIZE = 110

# Template capture: radius of the inner-disc (needle area) to mask out (px).
# Measured from live GitzWC capture: inner disc edge sits at ~38 px from centre.
INNER_DISC_RADIUS = 38

# Outer radius of the bezel ring in the mask (px from centre).
# Pixels beyond this radius (corners, title-bar strips, background) are excluded.
# Default: CAPTURE_SIZE // 2 - 1  (just inside the crop edge).
OUTER_DISC_RADIUS = CAPTURE_SIZE // 2 - 1  # 54 px for a 110-px crop


# ── RoseRegion dataclass ───────────────────────────────────────────────────

@dataclass
class RoseRegion:
    """Screen bounding box identifying where the wind rose HUD is currently rendered."""
    x: int
    y: int
    w: int
    h: int
    match_score: float

    @property
    def center(self) -> tuple[int, int]:
        """Disc centre in absolute screen coordinates."""
        return (
            self.x + self.w // 2 + CENTER_OFFSET_X,
            self.y + self.h // 2 + CENTER_OFFSET_Y,
        )

    @property
    def mss_region(self) -> dict:
        """mss grab dict for this bounding box."""
        return {"left": self.x, "top": self.y, "width": self.w, "height": self.h}


# ── Template loading ───────────────────────────────────────────────────────

def _load_templates() -> tuple:
    """Load template and mask PNGs as grayscale numpy arrays.

    Returns:
        (tmpl, mask) — both np.ndarray, dtype uint8, shape (H, W).

    Raises SystemExit(1) if either file is missing.
    """
    if not TEMPLATE_PATH.exists():
        print(
            f"ERROR: Template not found at {TEMPLATE_PATH}\n"
            "       Run: python tools/wind_reader_gitzwc.py --capture-template"
        )
        sys.exit(1)
    if not MASK_PATH.exists():
        print(
            f"ERROR: Mask not found at {MASK_PATH}\n"
            "       Run: python tools/wind_reader_gitzwc.py --capture-template"
        )
        sys.exit(1)

    tmpl = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(MASK_PATH),     cv2.IMREAD_GRAYSCALE)

    if tmpl is None:
        print(f"ERROR: Could not read {TEMPLATE_PATH} (corrupt file?)")
        sys.exit(1)
    if mask is None:
        print(f"ERROR: Could not read {MASK_PATH} (corrupt file?)")
        sys.exit(1)

    return tmpl, mask


# ── Template capture ───────────────────────────────────────────────────────

def run_capture_template(monitor_idx: int = 2) -> None:
    """Interactive tool: capture a full-screen screenshot, let the player click
    the rose centre, crop a CAPTURE_SIZE × CAPTURE_SIZE region, auto-generate
    the ring mask, and save both PNGs to assets/.

    The mask is constructed as:
        - all-white (255) canvas  →  bezel ring pixels contribute to matchTemplate
        - filled black (0) circle at template centre  →  inner disc excluded
        - outer padding beyond the crop edge is already 0 by the white-only canvas
          but is cropped away; only the bounding-box matters for matchTemplate.
    Result: white annular bezel ring, black inner disc and outer padding.
    """
    print("GitzWC template capture")
    print("  Make sure GitzWC is open and the wind rose is visible.")
    print("  A full-screen window will open.")
    print("  Click the CENTRE of the wind rose disc, then press any key.\n")

    with mss_pkg.mss() as sct:
        monitor = sct.monitors[monitor_idx]
        raw = sct.grab(monitor)

    # mss returns BGRA; convert to BGR for cv2 display
    screen_bgra = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(
        (raw.height, raw.width, 4)
    )
    screen_bgr = cv2.cvtColor(screen_bgra, cv2.COLOR_BGRA2BGR)

    # ── Interactive click ────────────────────────────────────────────────
    clicked = []
    clean_bg = screen_bgr.copy()   # never modified — used to erase previous crosshair
    annotated = screen_bgr.copy()  # redrawn from clean_bg on each click

    win_name = "Click wind rose centre — any key to confirm"

    def _on_mouse(event, x, y, flags, param):  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            # Reset to clean background so the old crosshair disappears
            annotated[:] = clean_bg
            cv2.drawMarker(annotated, (x, y), (0, 255, 0),
                           cv2.MARKER_CROSS, markerSize=40, thickness=2)
            cv2.imshow(win_name, annotated)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Try to show at a reasonable size on common 2560×1440 monitors
    display_h, display_w = screen_bgr.shape[:2]
    cv2.resizeWindow(win_name, min(display_w, 1280), min(display_h, 720))

    cv2.setMouseCallback(win_name, _on_mouse)
    cv2.imshow(win_name, annotated)

    while True:
        key = cv2.waitKey(100)
        if key != -1 and len(clicked) > 0:
            break
        # Allow closing the window to abort
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Capture cancelled — window closed.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    cx_screen, cy_screen = clicked[-1]
    half = CAPTURE_SIZE // 2

    # ── Crop template ────────────────────────────────────────────────────
    x0 = max(cx_screen - half, 0)
    y0 = max(cy_screen - half, 0)
    x1 = x0 + CAPTURE_SIZE
    y1 = y0 + CAPTURE_SIZE

    # Guard against screen-edge clipping
    h_full, w_full = screen_bgr.shape[:2]
    if x1 > w_full or y1 > h_full:
        print(
            f"WARNING: Rose centre ({cx_screen}, {cy_screen}) is too close to the "
            "screen edge for a full crop.  Try clicking closer to the centre of the rose."
        )
        x1 = min(x1, w_full)
        y1 = min(y1, h_full)

    crop_bgr  = screen_bgr[y0:y1, x0:x1]
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # ── Build ring mask ──────────────────────────────────────────────────
    # Black canvas  → outer padding (corners, title-bar strips) excluded.
    # White circle  → outer ring band included in matchTemplate scoring.
    # Black circle  → inner needle area excluded.
    # Result: only the bezel ring band is white (active pixels).
    tmpl_cx = crop_gray.shape[1] // 2
    tmpl_cy = crop_gray.shape[0] // 2
    mask_canvas = np.zeros((CAPTURE_SIZE, CAPTURE_SIZE), dtype=np.uint8)
    cv2.circle(mask_canvas, (tmpl_cx, tmpl_cy), OUTER_DISC_RADIUS, 255, thickness=-1)
    cv2.circle(mask_canvas, (tmpl_cx, tmpl_cy), INNER_DISC_RADIUS, 0,   thickness=-1)

    # ── Save assets ──────────────────────────────────────────────────────
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(TEMPLATE_PATH), crop_gray)
    cv2.imwrite(str(MASK_PATH),     mask_canvas)

    print(f"Template saved → {TEMPLATE_PATH}  ({crop_gray.shape[1]}×{crop_gray.shape[0]} px)")
    print(f"Mask saved     → {MASK_PATH}  (inner disc radius={INNER_DISC_RADIUS} px)")
    print()
    print("Next steps:")
    print("  1. Run: python tools/wind_reader_gitzwc.py --debug  (verify bounding box)")
    print("  2. git add assets/wind_rose_template.png assets/wind_rose_mask.png")
    print("  3. git commit -m 'feat: add GitzWC wind rose template assets'")
    print("  4. Run: python tools/wind_reader_gitzwc.py  (start polling)")


# ── Screen capture ─────────────────────────────────────────────────────────

def capture_screen(sct, monitor_idx: int = 2) -> np.ndarray:
    """Grab the full monitor and return a BGRA numpy array."""
    monitor = sct.monitors[monitor_idx]
    raw = sct.grab(monitor)
    arr = np.frombuffer(raw.bgra, dtype=np.uint8).reshape((raw.height, raw.width, 4))
    return arr.copy()


def capture_rose_region(sct, region: RoseRegion) -> np.ndarray:
    """Grab only the cached RoseRegion bounding box and return a BGR numpy array."""
    raw = sct.grab(region.mss_region)
    arr = np.frombuffer(raw.bgra, dtype=np.uint8).reshape((raw.height, raw.width, 4))
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)


# ── Rose location ──────────────────────────────────────────────────────────

def locate_rose(
    screen_bgra: np.ndarray,
    tmpl: np.ndarray,
    mask: np.ndarray,
) -> "tuple[RoseRegion | None, float]":
    """Run full-screen matchTemplate (no mask — full template gives best score).

    Returns:
        (RoseRegion, score)  if score ≥ MATCH_THRESHOLD
        (None,       score)  otherwise

    The screen is converted to grayscale for a ~3× speedup over BGR matching.
    The mask parameter is accepted for API compatibility but not used in matching.
    """
    screen_gray = cv2.cvtColor(screen_bgra, cv2.COLOR_BGRA2GRAY)
    result = cv2.matchTemplate(screen_gray, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < MATCH_THRESHOLD:
        return None, float(max_val)

    return RoseRegion(
        x=max_loc[0],
        y=max_loc[1],
        w=tmpl.shape[1],
        h=tmpl.shape[0],
        match_score=float(max_val),
    ), float(max_val)


def run_scan_loop(sct, tmpl: np.ndarray, mask: np.ndarray, monitor_idx: int = 2) -> RoseRegion:
    """Poll until the rose is found; print [SCAN] progress each attempt."""
    while True:
        print("[SCAN] Searching for wind rose...", end=" ", flush=True)
        screen = capture_screen(sct, monitor_idx)
        region, score = locate_rose(screen, tmpl, mask)
        if region is not None:
            cx, cy = region.center
            print(f"found at ({cx}, {cy})  score={score:.3f}")
            return region
        print(f"not found (best score: {score:.3f}, need ≥ {MATCH_THRESHOLD:.2f}), retrying...")
        time.sleep(POLL_INTERVAL)


# ── Angle extraction ───────────────────────────────────────────────────────

def extract_angle(roi_bgr: np.ndarray) -> tuple:
    """Extract wind angle from a CAPTURE_SIZE×CAPTURE_SIZE BGR crop.

    The crop is centred exactly on the rose disc centre (RoseRegion.center),
    so the disc centre sits at approximately (CAPTURE_SIZE//2, CAPTURE_SIZE//2)
    within the array — matching ROSE_CENTER_X/Y from wind_reader.py, which
    were derived from the same fixed capture geometry.  No module patching needed.

    Returns:
        (angle_degrees: float, tip_pixel: tuple | None)
    """
    pil_img = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    return detect_direction_shape(pil_img)


# ── Debug annotation ───────────────────────────────────────────────────────

def _annotate_debug(
    screen_bgr: np.ndarray,
    region: RoseRegion,
    tip_pixel: "tuple | None",
) -> np.ndarray:
    """Return an annotated copy of the screen with bounding box and needle line."""
    out = screen_bgr.copy()
    # Green rectangle around detected bounding box
    cv2.rectangle(out, (region.x, region.y),
                  (region.x + region.w, region.y + region.h),
                  (0, 255, 0), 2)
    cx, cy = region.center
    # Centre dot
    cv2.circle(out, (cx, cy), 4, (0, 255, 0), -1)

    if tip_pixel is not None:
        # tip_pixel is relative to the ROI crop; convert to screen coords
        tip_screen = (region.x + tip_pixel[0], region.y + tip_pixel[1])
        cv2.line(out, (cx, cy), tip_screen, (0, 255, 255), 2)
        cv2.circle(out, tip_screen, 4, (0, 255, 255), -1)

    return out


# ── Save-screen diagnostic ────────────────────────────────────────────────

def run_save_screen(monitor_idx: int = 2) -> None:
    """Capture the current monitor and save it to data/ for manual inspection.

    Use this to verify that --monitor N is pointing at the correct screen
    before troubleshooting template-match failures.
    """
    with mss_pkg.mss() as sct:
        monitors = sct.monitors
        print(f"mss monitors: {len(monitors) - 1} physical monitor(s) detected")
        for i, m in enumerate(monitors):
            tag = "(combined)" if i == 0 else ("← scanning this one" if i == monitor_idx else "")
            print(f"  [{i}] {m['width']}×{m['height']} at ({m['left']},{m['top']})  {tag}")
        if monitor_idx >= len(monitors):
            print(f"ERROR: monitor {monitor_idx} does not exist.  Use --monitor 1 for the primary.")
            sys.exit(1)
        screen_bgra = capture_screen(sct, monitor_idx)

    screen_bgr = cv2.cvtColor(screen_bgra, cv2.COLOR_BGRA2BGR)
    out_path = DATA_DIR / "wind_screen_capture.png"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), screen_bgr)
    print(f"\nSaved → {out_path}  ({screen_bgr.shape[1]}×{screen_bgr.shape[0]} px)")
    print("Open this file to confirm the correct monitor is being captured.")
    if TEMPLATE_PATH.exists():
        tmpl = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
        print(f"Template: {tmpl.shape[1]}×{tmpl.shape[0]} px  (loaded OK)")
    else:
        print(f"Template: MISSING at {TEMPLATE_PATH}")


# ── Locate diagnostic ────────────────────────────────────────────────────

def run_locate(monitor_idx: int = 2) -> None:
    """Single scan ignoring MATCH_THRESHOLD: draw best-match box on a saved PNG.

    Green box  = score >= threshold (would be found normally).
    Red box    = score < threshold  (currently failing; check if box is on the rose).

    If the box is on the rose but red, just lower the threshold:
        python tools/wind_reader_gitzwc.py --threshold <score * 0.95>
    If the box is in the wrong place, re-run --capture-template.
    """
    tmpl, mask = _load_templates()

    with mss_pkg.mss() as sct:
        screen_bgra = capture_screen(sct, monitor_idx)

    screen_gray = cv2.cvtColor(screen_bgra, cv2.COLOR_BGRA2GRAY)
    result = cv2.matchTemplate(screen_gray, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    screen_bgr = cv2.cvtColor(screen_bgra, cv2.COLOR_BGRA2BGR)
    x, y = max_loc
    th, tw = tmpl.shape[:2]
    passed = max_val >= MATCH_THRESHOLD
    color = (0, 200, 0) if passed else (0, 0, 255)  # green = pass, red = fail

    cv2.rectangle(screen_bgr, (x, y), (x + tw, y + th), color, 3)
    label = f"score={max_val:.3f}  thresh={MATCH_THRESHOLD:.2f}  {'OK' if passed else 'FAIL'}"
    cv2.putText(screen_bgr, label, (x, max(y - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    out_path = DATA_DIR / "wind_locate_debug.png"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), screen_bgr)

    status = "PASS" if passed else "FAIL"
    print(f"Best match : score={max_val:.3f} at ({x}, {y})  [{status}]")
    print(f"Threshold  : {MATCH_THRESHOLD:.2f}")
    print(f"Saved      \u2192 {out_path}")
    if not passed:
        suggested = round(max_val * 0.93, 2)
        print(f"\nIf the red box is ON the wind rose, lower the threshold:")
        print(f"  python tools/wind_reader_gitzwc.py --threshold {suggested}")
        print(f"If the box is in the WRONG place, re-run: --capture-template")


# ── Debug mode ─────────────────────────────────────────────────────────────

def run_debug(monitor_idx: int = 2) -> None:
    """Single-frame mode: locate rose, annotate screenshot, save to data/wind_debug_gitzwc.png."""
    tmpl, mask = _load_templates()

    with mss_pkg.mss() as sct:
        region = run_scan_loop(sct, tmpl, mask, monitor_idx)
        roi_bgr = capture_rose_region(sct, region)

    angle, tip_pixel = extract_angle(roi_bgr)

    with mss_pkg.mss() as sct:
        screen_bgra = capture_screen(sct, monitor_idx)

    screen_bgr = cv2.cvtColor(screen_bgra, cv2.COLOR_BGRA2BGR)
    annotated  = _annotate_debug(screen_bgr, region, tip_pixel)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEBUG_IMG), annotated)

    cx, cy = region.center
    print(f"Region : ({region.x}, {region.y})  size {region.w}×{region.h}  "
          f"centre ({cx}, {cy})  score={region.match_score:.2f}")
    print(f"Angle  : {angle:.1f}°")
    print(f"Debug  : {DEBUG_IMG}")


# ── Main poll loop ─────────────────────────────────────────────────────────

def run_reader(monitor_idx: int = 2) -> None:
    """Continuous poll loop: locate rose, read angle every POLL_INTERVAL seconds,
    write data/wind.json.  Re-scans after RESCAN_MISS_LIMIT consecutive low-score frames.
    """
    tmpl, mask = _load_templates()

    print(f"GitzWC wind reader started  →  {WIND_JSON}")
    print(f"Template: {TEMPLATE_PATH} ({tmpl.shape[1]}×{tmpl.shape[0]})")
    print("Ctrl+C to stop.\n")

    buf = WindBuffer(size=BUFFER_SIZE)
    last_good_angle: float = 0.0
    miss_counter: int = 0

    with mss_pkg.mss() as sct:
        region = run_scan_loop(sct, tmpl, mask, monitor_idx)

        while True:
            try:
                # ── Per-poll cached re-check ───────────────────────────────
                roi_bgr    = capture_rose_region(sct, region)
                roi_gray   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                check      = cv2.matchTemplate(roi_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(check)

                if score < TRACK_THRESHOLD:
                    miss_counter += 1
                    if miss_counter >= RESCAN_MISS_LIMIT:
                        write_wind(strength=None, angle=last_good_angle, stable=False)
                        print(
                            f"\n[LOST] Rose not found ({miss_counter} misses). Re-scanning..."
                        )
                        buf.clear()
                        miss_counter = 0
                        region = run_scan_loop(sct, tmpl, mask, monitor_idx)
                    time.sleep(POLL_INTERVAL)
                    continue

                miss_counter = 0

                # ── Angle extraction + buffer ──────────────────────────────
                raw_angle, _ = extract_angle(roi_bgr)
                buf.push(0, raw_angle)  # strength always 0/None for GitzWC reader

                if buf.full:
                    stable_angle = buf.stable_angle()
                    last_good_angle = stable_angle
                    write_wind(strength=None, angle=stable_angle, stable=True)
                    print(
                        f"\rWind: -- @ {stable_angle:>6.1f}°  [raw {raw_angle:.1f}°]    ",
                        end="", flush=True,
                    )
                else:
                    remaining = BUFFER_SIZE - len(buf._angles)
                    print(
                        f"\rBuffering… ({remaining} frames left)   ",
                        end="", flush=True,
                    )

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"\n[error] {exc}")

            time.sleep(POLL_INTERVAL)


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    global MATCH_THRESHOLD

    parser = argparse.ArgumentParser(
        description="GitzWC dynamic-position wind rose reader"
    )
    parser.add_argument(
        "--capture-template", action="store_true",
        help="Interactive: capture full-screen screenshot, click rose centre, "
             "save assets/wind_rose_template.png + assets/wind_rose_mask.png, then exit.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Single frame: locate rose, save annotated debug image, then exit.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, metavar="SCORE",
        help=f"Override MATCH_THRESHOLD (default: {MATCH_THRESHOLD}).",
    )
    parser.add_argument(
        "--monitor", type=int, default=2, metavar="N",
        help="mss monitor index (1-based; default: 2 = secondary/main monitor).",
    )
    parser.add_argument(
        "--save-screen", action="store_true",
        help="Save the current monitor capture to data/wind_screen_capture.png, "
             "list detected monitors, and exit.  Use this to verify --monitor N "
             "is pointing at the correct screen.",
    )
    parser.add_argument(
        "--locate", action="store_true",
        help="Single scan ignoring threshold: draw best-match bounding box on "
             "data/wind_locate_debug.png and print the score.  "
             "Use this to check if the template is finding the right place.",
    )
    args = parser.parse_args()

    # Apply runtime overrides
    if args.threshold is not None:
        MATCH_THRESHOLD = args.threshold

    try:
        if args.capture_template:
            run_capture_template(monitor_idx=args.monitor)
        elif args.save_screen:
            run_save_screen(monitor_idx=args.monitor)
        elif args.locate:
            run_locate(monitor_idx=args.monitor)
        elif args.debug:
            run_debug(monitor_idx=args.monitor)
        else:
            run_reader(monitor_idx=args.monitor)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
