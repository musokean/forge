"""等待动画：AI 回复等待期的旋转指示器（spinner），防止用户以为卡死。

终端原地刷新（\\r），不产生换行；退出时清行，不污染后续输出。

用法（异步上下文）：
    async with spinner("💭 思考中"):
        result = await slow_call()

或手动（配合流式：首字到达时停掉）：
    t = spinner_start("💭 思考中")
    ... await 首字 ...
    spinner_stop(t)
"""
import asyncio
import sys

from .console import C, paint

FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Spinner:
    """旋转指示器：进入上下文启动，退出时停止并清行。"""

    def __init__(self, message: str = "", interval: float = 0.1):
        self.message = message
        self.interval = interval
        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        self._task = spinner_start(self.message, self.interval)
        return self

    async def __aexit__(self, *exc):
        spinner_stop(self._task)
        return False


def spinner_start(message: str = "", interval: float = 0.1) -> asyncio.Task:
    """启动旋转动画（后台 task），返回 task 句柄供 spinner_stop 停止。"""
    async def _run():
        i = 0
        try:
            while True:
                frame = FRAMES[i % len(FRAMES)]
                line = f"{frame} {message}…" if message else frame
                sys.stdout.write("\r" + paint(line, C.SKY_DIM))
                sys.stdout.flush()
                i += 1
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise

    return asyncio.create_task(_run())


def spinner_stop(task: asyncio.Task | None):
    """停止动画并清掉当前行（不残留字符）。"""
    if task is None:
        return
    task.cancel()
    try:
        # 给 task 一个让出机会完成清理，避免 CancelledError 警告
        pass
    except Exception:
        pass
    sys.stdout.write("\r" + " " * 32 + "\r")
    sys.stdout.flush()
