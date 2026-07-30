"""Borderless always-on-top picture-in-picture window."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

import cv2
import numpy as np


class PictureInPictureWindow:
    """OpenCV window anchored to the desktop work area's bottom-right."""

    def __init__(
        self,
        title: str,
        *,
        width: int = 480,
        height: int = 270,
        margin: int = 18,
    ) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.margin = margin
        self._configured = False

    def create(self) -> None:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.title, self.width, self.height)
        # Materialize the native window before looking up its Win32 handle.
        cv2.imshow(self.title, np.zeros((self.height, self.width, 3), dtype=np.uint8))
        cv2.waitKey(1)
        self._configure()

    def show(self, frame: np.ndarray) -> None:
        display = cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow(self.title, display)
        if not self._configured:
            self._configure()

    def _configure(self) -> None:
        if sys.platform != "win32":
            cv2.moveWindow(self.title, 20, 20)
            try:
                cv2.setWindowProperty(self.title, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
            self._configured = True
            return

        user32 = ctypes.windll.user32
        user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user32.FindWindowW.restype = wintypes.HWND
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_long,
        ]
        user32.SetWindowLongW.restype = ctypes.c_long
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.SystemParametersInfoW.argtypes = [
            wintypes.UINT,
            wintypes.UINT,
            wintypes.LPVOID,
            wintypes.UINT,
        ]
        user32.SystemParametersInfoW.restype = wintypes.BOOL
        hwnd = user32.FindWindowW(None, self.title)
        if not hwnd:
            return

        # Remove title bar, resize frame, and taskbar presence.
        gwl_style = -16
        gwl_exstyle = -20
        ws_popup = 0x80000000
        ws_caption = 0x00C00000
        ws_thickframe = 0x00040000
        ws_sysmenu = 0x00080000
        ws_minimizebox = 0x00020000
        ws_maximizebox = 0x00010000
        ws_ex_toolwindow = 0x00000080

        style = user32.GetWindowLongW(hwnd, gwl_style)
        style &= ~(
            ws_caption
            | ws_thickframe
            | ws_sysmenu
            | ws_minimizebox
            | ws_maximizebox
        )
        style |= ws_popup
        user32.SetWindowLongW(hwnd, gwl_style, style)

        exstyle = user32.GetWindowLongW(hwnd, gwl_exstyle)
        user32.SetWindowLongW(hwnd, gwl_exstyle, exstyle | ws_ex_toolwindow)

        rect = wintypes.RECT()
        spi_getworkarea = 0x0030
        if not user32.SystemParametersInfoW(
            spi_getworkarea, 0, ctypes.byref(rect), 0
        ):
            rect.left = 0
            rect.top = 0
            rect.right = user32.GetSystemMetrics(0)
            rect.bottom = user32.GetSystemMetrics(1)

        x = rect.right - self.width - self.margin
        y = rect.bottom - self.height - self.margin

        hwnd_topmost = wintypes.HWND(-1)
        swp_noactivate = 0x0010
        swp_framechanged = 0x0020
        swp_showwindow = 0x0040
        user32.SetWindowPos(
            hwnd,
            hwnd_topmost,
            x,
            y,
            self.width,
            self.height,
            swp_noactivate | swp_framechanged | swp_showwindow,
        )
        self._configured = True
