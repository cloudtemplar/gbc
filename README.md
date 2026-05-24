# GunBound Classic Shot Calculator

A physics-simulation + self-learning shot calculator for GunBound Classic. Given the current wind and distance, it suggests angle/power combinations, learns from your real landing results, and refines its model over time.

---

## Quick Start

> **Run as Administrator** when playing GitzWC (or any version that launches elevated).
> Windows blocks hotkey detection from non-elevated processes when an elevated game window has focus.

```
python main.py
```

To record known shots without going through the solver (e.g. from a reference table):

```
python main.py --training
```

For the on-screen ruler, see [Ruler Overlay](#ruler-overlay) below.

---

You will be prompted step by step:

```
=== GunBound Shot Calculator ===
  Calibrated mobiles: armor, ice
  Training shots loaded: 104

Mobile: armor
Target SD (0.1–3.0): 0.875
Wind strength (0–26): 5
Wind angle (0=up, 90=toward enemy, -90=away, ±180=down): 70
Height diff (positive = you are higher than enemy, -1.0 to 1.0): 0

  Suggestions for ARMOR @ 0.875 SD  (wind 5.0@+70°  height +0.00):
    1) angle=  70°  power=2.55  [data: 1 shot(s), ±0.000 SD spread]
    2) angle=  56°  power=2.24  [physics, err +0.0008 SD]
    3) angle=  68°  power=2.53  [physics, err -0.0012 SD]

  Which did you use? (1–5 or Enter to skip): 1
  Where did it land (SD)? 0.87
  Recorded. (105 total shots)
```

Every 5 recorded shots the model auto-recalibrates for that mobile.

---

## Commands

| Command | Description |
|---|---|
| `python main.py` | Interactive calculator |
| `python main.py --training` | Record shots without the solver |
| `python main.py --calibrate` | Recalibrate all mobiles from `data/training_data.json` |
| `python main.py --validate` | Print per-shot errors and MAE for all training shots |
| `python scripts/import_baseline.py` | Bulk-import reference shots into training data |
| `python tools/ruler.py` | Launch on-screen SD ruler overlay |
| `python tools/gen_ruler.py` | Regenerate `assets/ruler.png` (run after ruler config changes) |
| `python tools/wind_reader_gitzwc.py` | Run GitzWC wind reader standalone |
| `python tools/wind_reader_gitzwc.py --capture-template` | (Re)capture wind rose template |
| `python main.py --wind-reader gitzwc` | Start calculator with GitzWC wind reader pre-selected |

---

## Input Reference

| Field | Range | Description |
|---|---|---|
| `mobile` | see [Supported Mobiles](#supported-mobiles) | The mobile (tank) you are using |
| `sd` | 0.1 – 3.0 | Screen Distance to the target |
| `wind_strength` | 0 – 26 | Wind bar value shown in-game |
| `wind_angle` | -180 – 180 | Wind direction in degrees (see below) |
| `height_diff` | -1.0 – 1.0 | Signed vertical offset: positive = you are higher than the enemy. Default 0 |

### Wind angle convention

```
       0° = straight up     (no horizontal effect)
      90° = toward enemy    (pushes projectile toward the enemy)
     -90° = away from enemy (pushes projectile away from the enemy)
  ±180°  = straight down   (no horizontal effect)
```

> **If the enemy is to the right:** enter the wind angle exactly as shown on the in-game compass.
> **If the enemy is to the left:** flip the sign of the angle (e.g. compass shows 90°, enter -90°; compass shows -45°, enter +45°). Purely vertical wind (0° or ±180°) does not need flipping.

---

## Supported Mobiles

All 18 mobiles are supported. Gravity is fixed from reverse-engineered engine data. Wind coefficients are calibrated from training shots — mobiles with fewer than 3 recorded shots use defaults derived from Armor's calibration.

```
aduka  armor  asate  bigfoot  boomer  dragon  grub  ice  jd  jfrog
kalsiddon  knight  lightning  mage  nak  raon  trico  turtle
```

Currently calibrated from real shots: **armor**, **ice**.

---

## How It Works

### 1. Physics simulation — `simulate_shot()`

**Initial velocity** ($v_\text{init} = \text{power}^{e} \cdot s$, where $e$ = `power_exp`, $s$ = `v_scale`):

$$v_x^0 = v_\text{init} \cdot \cos\alpha, \quad v_y^0 = v_\text{init} \cdot \sin\alpha$$


**Wind decomposition** (integer-truncated before coefficient scaling, matching the engine):

$$w_x = \lfloor W \sin\theta \rfloor \cdot c_x, \quad w_y = \lfloor W \cos\theta \rfloor \cdot c_y$$

**Euler integration** (dt = 0.05, position-before-velocity, stop when y ≤ 0 on descent):
```
x  += vx * dt
y  += vy * dt
vx += wx * dt
vy += (wy - g) * dt
```

A linear interpolation at the y = 0 crossing eliminates dt overshoot, critical for height_diff < 0. Per-mobile gravity $g$ is fixed from the engine's internal gravity table (Armor = 9.8 SD/step²; all others scaled by ratio).

---

### 2. Suggestions — `suggest_shots()`

Two-tier hybrid approach:

**Tier 1 — data match** (highest confidence): if any training shots have a similarity score $d_\text{sim} < 0.08$ to the current inputs, the most-sampled angle/power cluster is returned directly. $d_\text{sim}$ combines SD, wind components, and height, normalised to [0, 1].

**Tier 2 — physics solver**: a coarse→refine angle sweep (35°–89°, binary power search per angle). The solver target is residual-corrected — the mean signed error over nearby training shots is added to the target SD before solving, so systematic model bias is neutralised.

Up to 5 suggestions are returned, biased toward the 45–80° practical range, with at least 3° separation between any two suggestions.

---

### 3. Calibration — `calibrate()` / `recalibrate_all()`

Auto-recalibration runs every 5 recorded shots. Three-phase fitting per mobile:

**Phase A** — fit `(v_scale, power_exp)` from no-wind shots only. No-wind shots isolate these two parameters from wind-coefficient degeneracy.

**Phase B** — fix `(v_scale, power_exp)`, fit `(wind_x_coeff, wind_y_coeff)` from all shots.

**Phase C** — joint coordinate-descent refinement of all four parameters.

Loss function (distance-weighted squared error):
$$\mathcal{L} = \sum_{i} \frac{(\hat{x}_i - x_i)^2}{0.5 + |x_i|}$$

Fitted parameters are written to `config/mobiles_v2.json`. Run `python main.py --validate` to see MAE per mobile after calibration.

---

## Project Structure

```
gunbound/
├── src/gunbound/          # core package
│   ├── constants.py       # MOBILE_PHYSICS, solver params, thresholds
│   ├── models.py          # ShotResult dataclass
│   ├── physics.py         # effective_gravity, wind_components, simulate_shot
│   ├── calibration.py     # calibrate, recalibrate_all, validate
│   ├── solver.py          # solve (coarse→refine)
│   ├── matching.py        # suggest_shots, find_similar_shots, clustering
│   ├── storage.py         # load/save mobiles & training data; path constants
│   ├── position_capture.py # mouse hotkey capture (optional, needs pynput)
│   └── cli.py             # main(), training_mode()
├── config/
│   └── mobiles_v2.json    # calibrated parameters per mobile
├── data/
│   └── training_data.json # all recorded shots (never delete entries)
├── assets/
│   └── ruler.png          # pre-rendered ruler image
├── docs/                  # physics.md, data_collection.md, memory_analysis.md
├── legacy/
│   └── calc_legacy.py     # v1 reference (band-based, kept for comparison)
├── tools/
│   ├── ruler.py           # on-screen SD ruler overlay
│   ├── gen_ruler.py       # regenerates assets/ruler.png
│   ├── wind_reader.py     # classic GunBound wind reader (fixed HUD position)
│   ├── wind_reader_gitzwc.py  # GitzWC wind reader — auto-detects wind rose on screen
│   └── memory_reader.py   # process memory reader (research tool)
├── scripts/
│   └── import_baseline.py # bulk-import reference shots into training data
├── tests/
├── main.py                # entrypoint
├── pyproject.toml
└── requirements.txt
```

---

## Data Files

| File | Purpose |
|---|---|
| `config/mobiles_v2.json` | Per-mobile calibrated parameters. Written by `--calibrate`. |
| `data/training_data.json` | All recorded shots. Grows over time. **Never delete entries manually.** |

### `config/mobiles_v2.json` entry structure

```json
"armor": {
  "v_scale": 1.31502,
  "power_exp": 0.97071,
  "wind_x_coeff": 0.09710,
  "wind_y_coeff": 0.11745
}
```

### `data/training_data.json` entry structure

```json
{
  "mobile": "armor",
  "angle": 70,
  "power": 2.55,
  "wind_strength": 0.0,
  "wind_angle": 0.0,
  "height_diff": 0.0,
  "actual_sd": 0.875
}
```

---

## How to Use the Calculator Effectively

### Step 1 — Read your inputs correctly

- **SD:** read from the targeting indicator (use the ruler overlay for precision)
- **Wind strength:** the numeric value (0–26) shown in-game
- **Wind angle:** read from the in-game compass. Flip the sign if the enemy is to the left (see [Wind angle convention](#wind-angle-convention)). Purely vertical wind never needs flipping.
- **Height diff:** positive if you are higher than the enemy, negative if lower; 0 if flat

### Step 2 — Interpret the suggestions

Up to 5 shots are returned. `[data: N shot(s)]` means the suggestion comes directly from matching training shots — trust these more. `[physics]` means a solver result — accurate once the model is calibrated, less so on a fresh mobile.

### Step 3 — Always record your result

After firing, enter the actual landing SD even on a miss. The model only improves from recorded data. A short is just as informative as a hit.

### Step 4 — Understand the calibration cycle

Every 5 recorded shots for a mobile, calibration reruns automatically. The updated coefficients are written to `config/mobiles_v2.json` immediately and used in the next query.

**In early sessions**, physics suggestions may be off on uncalibrated mobiles — this is expected. Accuracy improves as data accumulates.

### Step 5 — What to do when shots are consistently wrong

| Symptom | Likely cause | Action |
|---|---|---|
| Always lands short | `v_scale` too low | Keep recording; calibration will raise it |
| Always lands long | `v_scale` too high | Keep recording |
| Wind pushes too much | `wind_x_coeff` too high | Keep recording |
| Wind pushes too little | `wind_x_coeff` too low | Keep recording |
| Suggestions are random-looking | Too few shots | Need ≥ 3 shots for calibration to run |

### Tips for faster calibration

- **Vary wind conditions.** Shots recorded only at zero wind can't calibrate wind coefficients.
- **Be consistent with SD reading.** Inconsistent readings are the main source of noise.
- **Don't record bad inputs.** If you mis-read the wind and the shot landed wildly off, skip recording that sample.

---

## Wind Reader (GitzWC)

`tools/wind_reader_gitzwc.py` automatically locates and reads the wind rose on screen, then writes angle and stability to `data/wind.json`, which the calculator polls to auto-fill the wind fields.

### Requirements

```
pip install mss opencv-python numpy
```

### One-time setup — capture template

With GitzWC open and the wind rose visible:

```
python tools/wind_reader_gitzwc.py --capture-template
```

Click the centre of the wind rose when prompted. The tool saves `assets/wind_rose_template.png`. Redo this only if the game resolution changes.

### Usage

The reader is launched automatically when you start `main.py`. On startup you will be prompted:

```
Wind reader:  [1] Classic (fixed position)  [2] GitzWC (auto-detect)
```

Choose **2** for GitzWC. To skip the prompt, pass the flag directly:

```
python main.py --wind-reader gitzwc
```

### Standalone diagnostics

| Command | What it does |
|---|---|
| `python tools/wind_reader_gitzwc.py` | Run the reader in the foreground |
| `python tools/wind_reader_gitzwc.py --locate` | Single scan — draws a box around the best match and saves `wind_locate_debug.png` |
| `python tools/wind_reader_gitzwc.py --debug` | Full annotated screenshot saved to `wind_debug_gitzwc.png` |
| `python tools/wind_reader_gitzwc.py --save-screen` | Save raw screen capture for verification |

### `data/wind.json` format

```json
{ "angle": 73, "strength": null, "stable": true, "ts": 1748123456.7 }
```

`stable: true` means the detected angle has been consistent across the last several frames. The calculator waits for stability before auto-filling.

---

## Ruler Overlay

`tools/ruler.py` is a transparent on-screen ruler you place over the GunBound window to read SD values accurately.

### Install dependency

```
pip install Pillow
pip install pywin32   # optional, for auto-snap to game window
```

### Run

```
python tools/ruler.py
```

The ruler auto-snaps to the screen center on startup. If `pywin32` is installed it will snap to the GunBound window.

### What it shows

| Element | Description |
|---|---|
| Horizontal strip (top) | SD scale 0 → 1.0 across the full width (1600px) |
| Vertical strip (left) | SD scale 0 → 0.75 down the full height (1200px) |
| Major ticks | Every 0.25 SD (every 400px) |
| Minor ticks | Every 0.125 SD (every 200px) |
| Dotted guide lines | Full-screen cross-hairs at every tick |

The background is fully transparent and click-through — the game remains playable underneath.

### Controls

| Control | Action |
|---|---|
| `R` button (top-right) | Re-snap to game window |
| `X` button (top-right) | Close |
| `Escape` | Close |
| Drag | Manually reposition |

### Configuration

At the top of `tools/ruler.py`:

```python
GAME_WINDOW_TITLE = "GunBound"   # substring of the game's window title (case-insensitive)
SCREEN_W = 1600                  # game client area width in pixels
SCREEN_H = 1200                  # game client area height in pixels
```

To regenerate `assets/ruler.png` after changing these values:

```
python tools/gen_ruler.py
```

---

## Mouse Position Hotkeys

When you and the enemy are both visible on screen at the same time, you can skip typing direction, SD, and height manually. The calculator reads them from your mouse cursor position in real time.

### How it works

1. **Press Ctrl+1** while hovering over your own character — marks your position.
2. **Press Ctrl+2** while hovering over the enemy — marks the target position.
3. **Switch back to the terminal** (or go to the next round) — the captured values are used automatically for the next query.

The horizontal distance between the two points maps to target SD (1 pixel = 1/1600 SD), and the vertical offset maps to height difference (200 pixels per slice = 0.125 SD). Shot direction (left/right) is inferred from which side the enemy was on.

Hotkeys respond **globally** — they work while the game window has focus. You do not need to alt-tab to the terminal to press them.

### Install dependency

```
pip install pynput
```

Or install all optional tools at once:

```
pip install -e ".[tools]"
```

If `pynput` is not installed, the calculator falls back to normal manual input with no other changes.

### Requirements

- GunBound must be running in **windowed mode** (not fullscreen) at its native resolution so that screen pixel coordinates match the ruler scale.
- The ruler overlay (`tools/ruler.py`) uses the same pixel-to-SD scale — if you resize the game window, re-run `tools/gen_ruler.py` and update `SCREEN_W`/`SCREEN_H` in `tools/ruler.py`.

### Hotkey bindings

| Hotkey | Action |
|---|---|
| `Ctrl+1` | Record own character position |
| `Ctrl+2` | Record target (enemy) position |

Pressing `Ctrl+1` again before completing a capture clears the previous pair and starts fresh.

### Fallback behaviour

If the captured positions fall outside the valid input range (SD < 0.1 or > 3.0; height outside ±1.0), the calculator prints a warning and falls back to manual prompts for that round. You can also simply not use the hotkeys and type all values as normal.

---

## Known Limitations

- Physics model has no drag or spin — accuracy depends entirely on fitted coefficients.
- Wind effect is linear; real GunBound wind may have nonlinear behaviour at obtuse angles (known open issue around θ ≈ 150°).
- No outlier rejection: one badly recorded shot can skew calibration.
- Solver sweep range is 35°–89°; shots requiring angles outside this range will not be suggested.
