"""
Low-level Windows mouse hook with conditional suppression.

Fires callbacks on Ctrl+xButton press/release and swallows the click so
the focused application (e.g. browser) doesn't see it. Other mouse events
pass through unchanged.

Why this is needed: pynput's mouse.Listener gets events after the OS has
already delivered them to the focused window, so a browser-back fires
before talktype intercepts. WH_MOUSE_LL runs in-process before delivery,
letting us return non-zero to swallow the event.
"""

import ctypes
import sys
import threading
import logging
from ctypes import wintypes

log = logging.getLogger("talktype.mouse_hook")

WH_MOUSE_LL = 14
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C
WM_QUIT = 0x0012
VK_CONTROL = 0x11
HC_ACTION = 0


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", wintypes.POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


LowLevelMouseProc = ctypes.WINFUNCTYPE(
    wintypes.LPARAM, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
)


if sys.platform == "win32":
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _user32.SetWindowsHookExW.restype = wintypes.HHOOK
    _user32.CallNextHookEx.restype = wintypes.LPARAM
else:
    _user32 = None
    _kernel32 = None


class CtrlMouseHook:
    """Hold Ctrl + xButton to fire push-to-talk callbacks; suppress the click.

    Usage:
        hook = CtrlMouseHook(x_button=1, on_press=start, on_release=stop)
        hook.start()
        ...
        hook.stop()

    The press is only registered (and suppressed) when Ctrl is held at the
    moment of click-down. The release is registered (and suppressed) only
    if the matching press was suppressed; otherwise it passes through so
    a normal mouse-back keeps working when Ctrl isn't held.
    """

    def __init__(self, x_button, on_press, on_release):
        if x_button not in (1, 2):
            raise ValueError("x_button must be 1 (back) or 2 (forward)")
        self._x_button = x_button
        self._on_press = on_press
        self._on_release = on_release
        self._hook = None
        self._thread_id = None
        self._thread = None
        self._proc = None
        self._recording = False

    def start(self):
        if sys.platform != "win32":
            log.warning("CtrlMouseHook only supported on Windows; not started")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread_id and _user32:
            _user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

    def _hook_proc(self, nCode, wParam, lParam):
        if nCode == HC_ACTION:
            data = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT))[0]
            x_button = (data.mouseData >> 16) & 0xFFFF

            if x_button == self._x_button:
                if wParam == WM_XBUTTONDOWN:
                    ctrl_held = bool(_user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)
                    if ctrl_held and not self._recording:
                        self._recording = True
                        try:
                            self._on_press()
                        except Exception as e:
                            log.error("on_press failed: %s", e)
                        return 1  # suppress click
                elif wParam == WM_XBUTTONUP:
                    if self._recording:
                        self._recording = False
                        try:
                            self._on_release()
                        except Exception as e:
                            log.error("on_release failed: %s", e)
                        return 1  # suppress click

        return _user32.CallNextHookEx(None, nCode, wParam, lParam)

    def _run(self):
        self._thread_id = _kernel32.GetCurrentThreadId()
        self._proc = LowLevelMouseProc(self._hook_proc)
        h_module = _kernel32.GetModuleHandleW(None)
        self._hook = _user32.SetWindowsHookExW(WH_MOUSE_LL, self._proc, h_module, 0)
        if not self._hook:
            log.error("SetWindowsHookExW failed: %d", ctypes.get_last_error())
            return

        msg = wintypes.MSG()
        while _user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) > 0:
            _user32.TranslateMessage(ctypes.byref(msg))
            _user32.DispatchMessageW(ctypes.byref(msg))

        _user32.UnhookWindowsHookEx(self._hook)
        self._hook = None
