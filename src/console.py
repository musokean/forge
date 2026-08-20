"""终端样式工具：ANSI 颜色 + 中文对齐画框，零依赖。

设 NO_COLOR=1 环境变量可禁用颜色（自动降级为纯文本）。
"""
import os
import sys


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    LIGHT_BLUE = "\033[94m"
    GRAY = "\033[90m"
    # 浅蓝主题变体（老大 2026-08-20：颜色统一浅蓝）
    SKY = "\033[94m"          # 同 LIGHT_BLUE，语义别名（成功/提示统一用）
    SKY_DIM = "\033[94;2m"    # 浅蓝暗淡（辅助文字/状态栏）
    SKY_BOLD = "\033[94;1m"   # 浅蓝粗体（强调）


# Windows 老终端需要激活 ANSI 转义（现代终端无害）
if os.name == "nt":
    try:
        os.system("")
    except Exception:
        pass

# Windows 控制台默认 GBK 编码，打印 emoji/特殊字符会 UnicodeEncodeError 静默崩溃；
# 强制 stdout/stderr 用 UTF-8 + replace 兜底（无法编码的字符替换为 ?，而非崩）
for _stream in (sys.stdout, sys.stderr):
    try:
        _enc = (_stream.encoding or "").lower().replace("-", "").replace("_", "")
        if _enc != "utf8":
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ENABLED = os.environ.get("NO_COLOR") is None


def paint(text, color):
    """给文本上色；NO_COLOR 时返回原文本。"""
    if not _ENABLED:
        return text
    return color + text + C.RESET


def display_width(s):
    """终端显示宽度：ASCII=1，中文等=2。"""
    return sum(2 if ord(ch) > 0x7f else 1 for ch in s)


def pad_display(s, width):
    """用空格把 s 补齐到指定显示宽度（中文按 2 列算）。"""
    return s + " " * (width - display_width(s))


def box(lines, color=C.CYAN):
    """给多行文本画框（中文对齐）。"""
    w = max(display_width(l) for l in lines)
    top = "╔" + "═" * (w + 2) + "╗"
    bottom = "╚" + "═" * (w + 2) + "╝"
    body = "\n".join("║ " + pad_display(l, w) + " ║" for l in lines)
    return paint(top + "\n" + body + "\n" + bottom, color)


def rule(width=44, char="─", color=C.GRAY):
    return paint(char * width, color)


def _terminal_width() -> int:
    """获取终端真实列宽（老大 2026-08-20：分隔线没到底——shutil 拿不到宽度时回退 80 导致线断在中途）。

    三级探测，取第一个有效值：
      1. Windows 原生 API（GetConsoleScreenBufferInfo，conhost 最准）
      2. COLUMNS 环境变量（Windows Terminal / PowerShell / Git Bash 都会设）
      3. shutil.get_terminal_size
      都失败回退 80。
    """
    # ① Windows 原生：conhost 真实宽度（最准）
    if os.name == "nt":
        try:
            import ctypes
            import struct
            buf = ctypes.create_string_buffer(22)  # CONSOLE_SCREEN_BUFFER_INFO
            if ctypes.windll.kernel32.GetConsoleScreenBufferInfo(
                    ctypes.windll.kernel32.GetStdHandle(-11), buf):
                # 结构布局：dwSize(4B) + dwCursorPosition(4B) + wAttributes(2B) = 偏移10 起是 srWindow
                # srWindow = { Left@10, Top@12, Right@14, Bottom@16 }（各 2B SHORT）
                left = struct.unpack_from("h", buf, 10)[0]
                right = struct.unpack_from("h", buf, 14)[0]
                w = right - left + 1
                if 20 <= w <= 500:
                    return w
        except Exception:
            pass
    # ② COLUMNS 环境变量（很多终端会设）
    try:
        cols = int(os.environ.get("COLUMNS", "0"))
        if 20 <= cols <= 500:
            return cols
    except (TypeError, ValueError):
        pass
    # ③ shutil
    try:
        import shutil
        cols = shutil.get_terminal_size().columns
        if cols > 0:
            return cols
    except Exception:
        pass
    return 80


def full_rule(char="─", color=C.LIGHT_BLUE, extra=0):
    """贯穿整个终端宽度的分隔线（老大 2026-08-20：分割线要一直到底/铺满全宽）。

    用**半角字符**（默认 ─）：1 字符恒占 1 列，直接铺满「终端列数 + extra」个——
    任何终端都绝对到头（extra 默认 0：老大从 +30 → +10 → 0 逐轮微调，最终定格铺满）。
    不用全角 ━：它在部分终端（conhost/某些字体）只占 1 列，按 2 列换算会少画一半。
    """
    cols = _terminal_width()
    n = max(10, cols + extra)
    return paint(char * n, color)
