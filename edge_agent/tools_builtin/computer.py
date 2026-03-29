"""Computer-use tools: screenshot, click, type, scroll, hotkey."""

from __future__ import annotations

import base64
import io
import logging

from edge_agent.tools import tool

log = logging.getLogger(__name__)

_DISPLAY_AVAILABLE = True
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
except Exception:
    _DISPLAY_AVAILABLE = False


def _check_display() -> str | None:
    if not _DISPLAY_AVAILABLE:
        return "pyautogui 不可用 (可能没有图形环境)"
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return "未检测到 DISPLAY 或 WAYLAND_DISPLAY 环境变量"
    return None


@tool("截取屏幕截图，返回 base64 编码的图像")
def screenshot() -> str:
    """Take a screenshot and return base64 PNG."""
    err = _check_display()
    if err:
        return err
    try:
        img = pyautogui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        w, h = img.size
        return f"截图成功 ({w}x{h}), base64 长度 {len(b64)}"
    except Exception as e:
        return f"截图失败: {e}"


@tool("点击屏幕指定坐标")
def click(x: int, y: int) -> str:
    err = _check_display()
    if err:
        return err
    try:
        pyautogui.click(x, y)
        return f"已点击 ({x}, {y})"
    except Exception as e:
        return f"点击失败: {e}"


@tool("在当前焦点位置输入文本")
def type_text(text: str) -> str:
    err = _check_display()
    if err:
        return err
    try:
        pyautogui.typewrite(text, interval=0.02) if text.isascii() else pyautogui.write(text)
        return f"已输入: {text[:50]}"
    except Exception as e:
        return f"输入失败: {e}"


@tool("滚动页面，direction 为 up 或 down")
def scroll(direction: str, amount: int = 3) -> str:
    err = _check_display()
    if err:
        return err
    try:
        clicks = amount if direction == "up" else -amount
        pyautogui.scroll(clicks)
        return f"已滚动 {direction} {amount} 格"
    except Exception as e:
        return f"滚动失败: {e}"


@tool("按下快捷键，如 'ctrl+c', 'alt+tab'")
def hotkey(keys: str) -> str:
    err = _check_display()
    if err:
        return err
    try:
        parts = [k.strip() for k in keys.split("+")]
        pyautogui.hotkey(*parts)
        return f"已按下: {keys}"
    except Exception as e:
        return f"快捷键失败: {e}"
