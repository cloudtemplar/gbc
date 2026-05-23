"""
constants.py — Project-wide constants for the GunBound shot calculator.

Covers:
  - Per-mobile physics data reverse-engineered from the game engine
  - Solver sweep parameters
  - Data-confidence thresholds for the training-data matching layer
"""

# ─────────────────────────────────────────────────────────────────────────────
# Per-mobile physics constants from reference reverse-engineering (memory.py)
#
# gravity:          game-internal downward acceleration (arbitrary units).
#                   Ratio matters — all mobiles are normalised relative to Armor.
# projectile_speed: used as wind-coefficient PRIOR for uncalibrated mobiles.
#                   In the reference it scales wind acceleration (not launch v).
# ─────────────────────────────────────────────────────────────────────────────
MOBILE_PHYSICS: dict[str, dict] = {
    "armor":     {"gravity": 73.5,  "projectile_speed": 0.740},
    "mage":      {"gravity": 71.5,  "projectile_speed": 0.780},
    "nak":       {"gravity": 93.0,  "projectile_speed": 0.990},
    "trico":     {"gravity": 84.0,  "projectile_speed": 0.870},
    "bigfoot":   {"gravity": 90.0,  "projectile_speed": 0.740},
    "boomer":    {"gravity": 62.5,  "projectile_speed": 1.395},
    "raon":      {"gravity": 81.0,  "projectile_speed": 0.827},
    "lightning": {"gravity": 65.0,  "projectile_speed": 0.720},
    "jd":        {"gravity": 62.5,  "projectile_speed": 0.625},
    "asate":     {"gravity": 76.0,  "projectile_speed": 0.765},
    "ice":       {"gravity": 62.5,  "projectile_speed": 0.625},
    "turtle":    {"gravity": 73.5,  "projectile_speed": 0.740},
    "grub":      {"gravity": 61.0,  "projectile_speed": 0.650},
    "aduka":     {"gravity": 65.5,  "projectile_speed": 0.695},
    "knight":    {"gravity": 65.5,  "projectile_speed": 0.695},
    "kalsiddon": {"gravity": 88.5,  "projectile_speed": 0.905},
    "jfrog":     {"gravity": 54.3,  "projectile_speed": 0.670},
    "dragon":    {"gravity": 54.3,  "projectile_speed": 0.670},
}

_ARMOR_G_REF = MOBILE_PHYSICS["armor"]["gravity"]   # 73.5  (normalisation anchor)
_G_BASE      = 9.8                                   # effective g for Armor in SD units

# ─────────────────────────────────────────────────────────────────────────────
# Per-mobile angle range overrides
# Keys not listed here use ANGLE_MIN / ANGLE_MAX as defaults.
# ─────────────────────────────────────────────────────────────────────────────
MOBILE_ANGLE_RANGE: dict[str, tuple[int, int]] = {
    "turtle":    (70, 89),
    "kalsiddon": (70, 89),
}

# ─────────────────────────────────────────────────────────────────────────────
# Solver parameters
# ─────────────────────────────────────────────────────────────────────────────
SOLVER_COARSE_STEP  = 2      # degrees, coarse angle sweep
SOLVER_REFINE_STEP  = 0.5    # degrees, refinement sweep
SOLVER_POWER_STEPS  = 40     # power grid points per angle
ANGLE_MIN           = 35
ANGLE_MAX           = 89
POWER_MIN           = 0.5
POWER_MAX           = 4.0
MAX_SUGGESTIONS     = 5
CLOSE_RANGE_THRESHOLD = 0.5  # SD below this → always include a high-angle suggestion
HIGH_ANGLE_MIN        = 80   # minimum angle considered "high" for close-range slot

# ─────────────────────────────────────────────────────────────────────────────
# Data-confidence thresholds (same scale as calc_legacy.py)
# ─────────────────────────────────────────────────────────────────────────────
MATCH_CLOSE_THRESHOLD = 0.08   # d_sim below this → direct data suggestion
MATCH_LOOSE_THRESHOLD = 0.15   # d_sim below this → residual correction applied

# ─────────────────────────────────────────────────────────────────────────────
# Position capture hotkeys (Win32 virtual-key codes for GetAsyncKeyState polling)
#
# The default Ctrl+1 / Ctrl+2 are claimed by GitzWC emotes (RegisterHotKey
# err 1409) and the game also uses DirectInput for char keys, blocking
# WH_KEYBOARD_LL hooks.  GetAsyncKeyState + different keys sidesteps both.
#
# To change: update the two _VK_* lines and the two *_LABEL strings.
# Common VK codes: F1–F12 = 0x70–0x7B  Insert=0x2D  Delete=0x2E
# ─────────────────────────────────────────────────────────────────────────────
_VK_CONTROL = 0x11
_VK_SHIFT   = 0x10
_VK_Z       = 0x5A
_VK_X       = 0x58

HOTKEY_OWN_VK       = (_VK_CONTROL, _VK_SHIFT, _VK_Z)   # Ctrl+Shift+Z → mark own position
HOTKEY_TARGET_VK    = (_VK_CONTROL, _VK_SHIFT, _VK_X)   # Ctrl+Shift+X → mark target position
HOTKEY_OWN_LABEL    = "Ctrl+Shift+Z"
HOTKEY_TARGET_LABEL = "Ctrl+Shift+X"

# ─────────────────────────────────────────────────────────────────────────────
# Known mobiles (derived from MOBILE_PHYSICS)
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_MOBILES: list[str] = sorted(MOBILE_PHYSICS.keys())
