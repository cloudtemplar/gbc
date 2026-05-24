"""
diag_hotkeys.py — Diagnose why Ctrl+1 / Ctrl+2 don't fire when the game window has focus.

Runs two parallel listeners:
  1. pynput GlobalHotKeys  — what the current position_capture.py uses
  2. pynput raw Listener   — logs EVERY key press/release so you can see if pynput
                             is even receiving events while the game is focused
  3. Win32 RegisterHotKey  — kernel-level message-queue hotkeys that bypass hooks

Press Ctrl+1 and Ctrl+2 with this console focused, then switch to the game window
and try the same keys.  Compare the output to see which method works.

Usage:
    python tools/diag_hotkeys.py

Press Ctrl+C to quit.
"""

from __future__ import annotations
import ctypes
import ctypes.wintypes
import threading
import time
import sys

# ── Win32 constants ───────────────────────────────────────────────────────────
MOD_NOREPEAT = 0x4000
MOD_CONTROL  = 0x0002
VK_1         = 0x31
VK_2         = 0x32
WM_HOTKEY    = 0x0312

user32 = ctypes.windll.user32

# ── pynput ────────────────────────────────────────────────────────────────────
try:
    from pynput import keyboard as _kb
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    print("[WARN] pynput not installed — skipping pynput tests")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ── 1. pynput GlobalHotKeys ───────────────────────────────────────────────────

def start_global_hotkeys():
    if not HAS_PYNPUT:
        return
    def _h1():
        print(f"[{_ts()}] pynput GlobalHotKeys  → Ctrl+1 FIRED")
    def _h2():
        print(f"[{_ts()}] pynput GlobalHotKeys  → Ctrl+2 FIRED")
    gh = _kb.GlobalHotKeys({"<ctrl>+1": _h1, "<ctrl>+2": _h2})
    gh.daemon = True
    gh.start()
    print("[INFO] pynput GlobalHotKeys listener started")


# ── 2. pynput raw Listener (logs all keys) ────────────────────────────────────

def start_raw_listener():
    if not HAS_PYNPUT:
        return

    ctrl_held = False

    def on_press(key):
        nonlocal ctrl_held
        try:
            name = key.char if hasattr(key, "char") else str(key)
        except Exception:
            name = str(key)

        if key in (_kb.Key.ctrl_l, _kb.Key.ctrl_r, _kb.Key.ctrl):
            ctrl_held = True

        # Highlight Ctrl+1 / Ctrl+2 specifically
        if ctrl_held and hasattr(key, "char") and key.char in ("1", "2"):
            print(f"[{_ts()}] pynput raw Listener  → Ctrl+{key.char} SEEN")
        else:
            print(f"[{_ts()}] pynput raw Listener  → key down: {name}")

    def on_release(key):
        nonlocal ctrl_held
        if key in (_kb.Key.ctrl_l, _kb.Key.ctrl_r, _kb.Key.ctrl):
            ctrl_held = False

    lst = _kb.Listener(on_press=on_press, on_release=on_release)
    lst.daemon = True
    lst.start()
    print("[INFO] pynput raw Listener started (logs ALL keys)")


# ── 3. Win32 RegisterHotKey ───────────────────────────────────────────────────

class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd",    ctypes.wintypes.HWND),
        ("message", ctypes.c_uint),
        ("wParam",  ctypes.wintypes.WPARAM),
        ("lParam",  ctypes.wintypes.LPARAM),
        ("time",    ctypes.c_uint32),
        ("pt",      ctypes.wintypes.POINT),
    ]


def start_register_hotkey():
    """
    Uses Win32 RegisterHotKey — sends WM_HOTKEY to *this thread's* message queue.
    Works even when games block WH_KEYBOARD_LL hooks because it bypasses hook chains.
    The downside: requires a GetMessage loop on a dedicated thread.
    """

    _HOTKEY_ID_1 = 100
    _HOTKEY_ID_2 = 101

    def _loop():
        # RegisterHotKey must be called from the SAME thread that runs GetMessage
        ok1 = user32.RegisterHotKey(None, _HOTKEY_ID_1, MOD_CONTROL | MOD_NOREPEAT, VK_1)
        ok2 = user32.RegisterHotKey(None, _HOTKEY_ID_2, MOD_CONTROL | MOD_NOREPEAT, VK_2)
        if not ok1:
            print(f"[WARN] RegisterHotKey Ctrl+1 failed (err {ctypes.GetLastError()}) — "
                  "another app may have claimed it")
        if not ok2:
            print(f"[WARN] RegisterHotKey Ctrl+2 failed (err {ctypes.GetLastError()}) — "
                  "another app may have claimed it")
        if ok1 or ok2:
            print("[INFO] Win32 RegisterHotKey listener started")

        msg = _MSG()
        while True:
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret == 0 or ret == -1:
                break
            if msg.message == WM_HOTKEY:
                if msg.wParam == _HOTKEY_ID_1:
                    print(f"[{_ts()}] Win32 RegisterHotKey → Ctrl+1 FIRED")
                elif msg.wParam == _HOTKEY_ID_2:
                    print(f"[{_ts()}] Win32 RegisterHotKey → Ctrl+2 FIRED")
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        user32.UnregisterHotKey(None, _HOTKEY_ID_1)
        user32.UnregisterHotKey(None, _HOTKEY_ID_2)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ── 4. GetAsyncKeyState polling ───────────────────────────────────────────────
# This is what position_capture.py now uses.  Reads directly from the kernel
# VK state table — independent of hooks and message queues.

_GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState

VK_CONTROL = 0x11
VK_SHIFT   = 0x10
VK_Z       = 0x5A
VK_X       = 0x58


def _key_down(vk: int) -> bool:
    return bool(_GetAsyncKeyState(vk) & 0x8000)


def start_async_poll():
    own_prev    = False
    target_prev = False

    def _poll():
        nonlocal own_prev, target_prev
        while True:
            own    = _key_down(VK_CONTROL) and _key_down(VK_SHIFT) and _key_down(VK_Z)
            target = _key_down(VK_CONTROL) and _key_down(VK_SHIFT) and _key_down(VK_X)

            if own and not own_prev:
                print(f"[{_ts()}] GetAsyncKeyState     → Ctrl+Shift+Z FIRED")
            if target and not target_prev:
                print(f"[{_ts()}] GetAsyncKeyState     → Ctrl+Shift+X FIRED")

            own_prev    = own
            target_prev = target
            time.sleep(0.05)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    print("[INFO] GetAsyncKeyState poll started (Ctrl+Shift+Z / Ctrl+Shift+X)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Hotkey diagnostics")
    print("=" * 60)
    print()
    print("Step 1: press Ctrl+Shift+Z and Ctrl+Shift+X with THIS console focused.")
    print("Step 2: switch to the GitzWC game window and repeat.")
    print("Step 3: compare which methods report FIRED vs silence.")
    print()
    print("Legend:")
    print("  pynput GlobalHotKeys  = hook-based (blocked by game DirectInput)")
    print("  pynput raw Listener   = raw hook — tells if pynput sees events at all")
    print("  Win32 RegisterHotKey  = message-queue (game has claimed Ctrl+1/2)")
    print("  GetAsyncKeyState      = kernel VK table poll — what cli.py now uses")
    print()

    start_global_hotkeys()
    start_raw_listener()
    start_register_hotkey()
    start_async_poll()

    print()
    print("All listeners running. Press Ctrl+C to quit.")
    print("-" * 60)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDone.")
        sys.exit(0)


if __name__ == "__main__":
    main()
