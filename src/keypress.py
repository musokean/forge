"""流式生成期间的键盘轮询（跨平台非阻塞）：Esc 打断 / 引导输入。

背景（老大需求 2026-08-20）：forge 思考/生成时不能打断、不能引导，只能干等。
这里提供两个能力：
  · poll_key()       非阻塞检测按键——流式循环里每收到一个 chunk 轮询一次，
                     返回 "ESC"（用户按 Esc 想中断）/ 其他按键字符 / None（无按键）
  · read_guide_line() 阻塞读一行「引导输入」——打断后让用户输入一句话，
                     模型按这句话调整方向重新生成；空回车 = 不引导（纯中断）

跨平台实现：
  · Windows：msvcrt.kbhit()/getwch()（非阻塞，不 enter raw mode，最稳）
  · Unix：select.select(stdin, 0) + 普通 read（依赖终端是 tty，非 tty 直接返回 None）
"""
import sys

try:
    import msvcrt  # Windows only
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False
    msvcrt = None  # Linux 占位：让测试 patch("src.keypress.msvcrt") 有目标可打（CI #1 实测）


def poll_key():
    """非阻塞检测是否有按键；返回 "ESC" / 按下的字符 / None（无按键）。

    注意：只消费一个字符；功能键（方向键等）在 msvcrt 下会返回 \x00 或 \xe0 前缀，
    这里不处理（视为普通键，触发引导模式也无妨——引导行会重新读，安全）。
    """
    if _HAS_MSVCRT:
        try:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                return "ESC" if ch in ("\x1b",) else ch
        except Exception:
            return None
        return None
    # Unix 兜底：仅 tty 且可读时读一个字符
    try:
        import select
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return "ESC" if ch == "\x1b" else ch
    except Exception:
        pass
    return None


def read_guide_line(prompt="  ✍ 引导 forge（直接回车不引导）› "):
    """阻塞读一行引导输入；空回车 / 中断返回 None。"""
    try:
        from .console import C, paint
        v = input(paint(prompt, C.SKY)).strip()
        return v or None
    except (EOFError, KeyboardInterrupt):
        return None
