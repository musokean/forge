"""键盘打断/引导测试：poll_key 跨平台降级 + agent 流式中断与引导重生成。

运行：python test_interrupt.py
依赖：managed venv（不联网——模型调用全 mock）
"""
import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, ".")

from src.keypress import poll_key, read_guide_line


# ===================== 1) poll_key 跨平台 =====================
class TestPollKey(unittest.TestCase):
    def test_returns_none_when_no_key(self):
        # 无 msvcrt 且 stdin 非 tty → None（不抛异常）
        with patch("src.keypress._HAS_MSVCRT", False), \
             patch("sys.stdin") as fake_stdin:
            fake_stdin.isatty.return_value = False
            self.assertIsNone(poll_key())

    def test_esc_on_windows(self):
        with patch("src.keypress._HAS_MSVCRT", True), \
             patch("src.keypress.msvcrt") as fake_m:
            fake_m.kbhit.return_value = True
            fake_m.getwch.return_value = "\x1b"
            self.assertEqual(poll_key(), "ESC")

    def test_plain_key_on_windows(self):
        with patch("src.keypress._HAS_MSVCRT", True), \
             patch("src.keypress.msvcrt") as fake_m:
            fake_m.kbhit.return_value = True
            fake_m.getwch.return_value = "a"
            self.assertEqual(poll_key(), "a")

    def test_no_key_on_windows(self):
        with patch("src.keypress._HAS_MSVCRT", True), \
             patch("src.keypress.msvcrt") as fake_m:
            fake_m.kbhit.return_value = False
            self.assertIsNone(poll_key())

    def test_msvcrt_exception_safe(self):
        with patch("src.keypress._HAS_MSVCRT", True), \
             patch("src.keypress.msvcrt") as fake_m:
            fake_m.kbhit.side_effect = OSError("no console")
            self.assertIsNone(poll_key())


# ===================== 2) read_guide_line =====================
class TestReadGuideLine(unittest.TestCase):
    def test_reads_line(self):
        with patch("builtins.input", return_value="用更简短的方式回答"):
            self.assertEqual(read_guide_line(), "用更简短的方式回答")

    def test_empty_returns_none(self):
        with patch("builtins.input", return_value="   "):
            self.assertIsNone(read_guide_line())

    def test_interrupt_returns_none(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            self.assertIsNone(read_guide_line())


# ===================== 3) agent 流式打断 =====================
def _mk_delta(content=None, reasoning=None, tool_calls=None):
    class _Fn:
        def __init__(self):
            self.name = None
            self.arguments = None
    class _TC:
        def __init__(self):
            self.index = 0
            self.id = None
            self.function = _Fn()
    class _D:
        pass
    d = _D()
    d.reasoning_content = reasoning
    d.content = content
    d.tool_calls = tool_calls or []
    return d


def _mk_stream(chunks, usage=None):
    """构造 stream_chat 的 async generator 返回值。"""
    async def gen():
        for c in chunks:
            yield c, usage
    return gen()


class TestStreamInterrupt(unittest.TestCase):
    def _mk_agent(self, show_spinner=True):
        """构造测试 Agent；打断相关测试用 show_spinner=True（打断检测挂在这个开关上）。"""
        from src.agent import Agent
        a = Agent.__new__(Agent)
        a._config_path = "config/models.yaml"
        a.cfg = {"models": [], "roles": {}}
        a.max_steps = 5
        a.max_context_tokens = 8000
        a.max_tool_output = 1500
        a.summary_ratio = 0.85
        a.approver = None
        a.role = "default"
        a._explicit_prompt = False
        a.stream = True
        a.show_spinner = show_spinner
        a.name = "forge"
        a.messages = [{"role": "system", "content": "sys"}]
        a.total_tokens = {"prompt": 0, "completion": 0}
        a._interrupts = 0
        from src.trace import Tracer
        a.tracer = Tracer(agent_name="forge", role="default", model="m")
        a._maybe_roll_summary = AsyncMock(return_value=None)
        a._maybe_reflect = AsyncMock(side_effect=lambda t, ans: ans)  # 直接返回原答案（AsyncMock 会 await）
        a._finish_trace = lambda *a_, **k: None
        a._print_status = lambda: None
        a._trace_llm = lambda *a_, **k: None
        a._add_usage = lambda u: None
        a._clip_tool_output = AsyncMock(side_effect=lambda t: t)  # AsyncMock 自动 await，返回原值
        a._execute_with_approval = lambda n, ar: {"ok": True, "data": "ok"}
        return a

    def test_esc_interrupt_returns_partial(self):
        """流式生成中按 Esc → 中断，返回已生成内容 + 标注。"""
        from src import agent as agent_mod
        a = self._mk_agent()
        # 第一个 chunk 无打断；第二个 chunk 到达时用户按 Esc → 中断
        seq = [iter([_mk_delta(content="已生成一半"), _mk_delta(content="的内容")])]
        def fake_stream(*args, **kwargs):
            return _mk_stream(seq[0])
        poll_results = iter([None, "ESC"])   # 首个 chunk 前不打断；第二个 chunk 前打断
        with patch.object(agent_mod, "stream_chat", new=fake_stream), \
             patch.object(agent_mod, "poll_key", side_effect=lambda: next(poll_results, None)), \
             patch.object(agent_mod, "resolve_model", return_value={"base_url": "http://x"}):
            out = asyncio.run(a.run("问题"))
        self.assertIn("已生成一半", out)
        self.assertIn("已中断", out)

    def test_guide_redirects_regeneration(self):
        """生成中按任意键 + 输入引导 → 重新生成，返回引导后的答案。"""
        from src import agent as agent_mod
        a = self._mk_agent()
        # 第一轮流式：产生部分内容后打断（poll 返回普通键 'a'）→ 引导
        # 第二轮流式：完整回答
        chunks1 = [_mk_delta(content="旧方向答案"), _mk_delta(content="旧")]
        chunks2 = [_mk_delta(content="引导后的完整答案")]
        streams = iter([chunks1, chunks2])
        def fake_stream(*args, **kwargs):
            return _mk_stream(next(streams))
        poll_results = iter(["a", None])   # 第一轮普通键 → 引导；第二轮无打断
        with patch.object(agent_mod, "stream_chat", new=fake_stream), \
             patch.object(agent_mod, "poll_key", side_effect=lambda: next(poll_results, None)), \
             patch.object(agent_mod, "read_guide_line", return_value="换个方向说"), \
             patch.object(agent_mod, "resolve_model", return_value={"base_url": "http://x"}):
            out = asyncio.run(a.run("问题"))
        self.assertIn("引导后的完整答案", out)
        # 引导被写进历史（作为 user 消息）
        user_msgs = [m["content"] for m in a.messages if m["role"] == "user"]
        self.assertTrue(any("换个方向说" in m for m in user_msgs))

    def test_three_interrupts_aborts(self):
        """连续打断 3 次 → 中止，不再循环。"""
        from src import agent as agent_mod
        a = self._mk_agent()
        chunks1 = [_mk_delta(content="x")]
        def fake_stream(*args, **kwargs):
            return _mk_stream(chunks1)
        # 每次都触发引导（永不结束）→ 第 3 次中止
        with patch.object(agent_mod, "stream_chat", new=fake_stream), \
             patch.object(agent_mod, "poll_key", return_value="a"), \
             patch.object(agent_mod, "read_guide_line", return_value="继续"), \
             patch.object(agent_mod, "resolve_model", return_value={"base_url": "http://x"}):
            out = asyncio.run(a.run("问题"))
        self.assertIn("中止", out)
        self.assertEqual(a._interrupts, 3)

    def test_explicit_prompt_no_interrupt(self):
        """辩论/并行专用人设（_explicit_prompt=True）不轮询键盘（不打断）。"""
        from src import agent as agent_mod
        a = self._mk_agent(show_spinner=True)
        a._explicit_prompt = True
        def fake_stream(*args, **kwargs):
            return _mk_stream([_mk_delta(content="完整答案")])
        with patch.object(agent_mod, "stream_chat", new=fake_stream), \
             patch.object(agent_mod, "poll_key") as pk:
            out = asyncio.run(a.run("问题"))
        self.assertIn("完整答案", out)
        pk.assert_not_called()  # 专用人设不轮询

    def test_no_spinner_no_interrupt(self):
        """show_spinner=False（并行子任务）不轮询键盘。"""
        from src import agent as agent_mod
        a = self._mk_agent(show_spinner=False)
        def fake_stream(*args, **kwargs):
            return _mk_stream([_mk_delta(content="后台答案")])
        with patch.object(agent_mod, "stream_chat", new=fake_stream), \
             patch.object(agent_mod, "poll_key") as pk:
            out = asyncio.run(a.run("问题"))
        self.assertIn("后台答案", out)
        pk.assert_not_called()


def _coro(v):
    async def _c():
        return v
    return _c()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
