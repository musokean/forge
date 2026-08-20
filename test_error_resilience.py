"""A07 错误兜底测试：模型调用失败（如 402 余额不足）不崩 REPL，返回友好提示。

覆盖三层：llm 降级（主角色失败→fallback 成功）、agent 友好返回（双通道全失败）、
agent 流式不崩。模拟的是账户级错误（402），非可重试错误。
"""
import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, ".")

from src.agent import Agent
from src.llm import chat, stream_chat


class Simulated402(Exception):
    """模拟 402 Insufficient Balance（账户级、不可重试的错误）。"""


def _mini_cfg():
    """最小配置：default→a（主）、fallback→b（降级），两个模型都有 key。"""
    return {
        "roles": {
            "default": {"model": "a", "label": "主"},
            "fallback": {"model": "b", "label": "降级"},
        },
        "models": {
            "a": {"base_url": "http://a/v1", "api_key": "k1", "model": "a-model", "label": "A"},
            "b": {"base_url": "http://b/v1", "api_key": "k2", "model": "b-model", "label": "B"},
        },
    }


class _FakeClient:
    """模拟 openai 客户端：主通道（stream=True）抛 402，非流式兜底成功。"""

    def __init__(self):
        self.n = 0
        resp = MagicMock()
        resp.choices[0].message.content = "降级通道兜底成功"
        resp.choices[0].message.reasoning_content = None
        resp.choices[0].message.tool_calls = None
        resp.usage = None
        self._resp = resp

    class _Completions:
        def __init__(self, owner):
            self._o = owner

        async def create(self, **kwargs):
            self._o.n += 1
            if kwargs.get("stream"):
                raise Simulated402("Insufficient Balance")
            return self._o._resp

    @property
    def chat(self):
        return MagicMock(completions=self._Completions(self))


class TestLlmDegrade(unittest.TestCase):
    """llm 层：主角色 402 → 自动降级 fallback。"""

    def test_stream_chat_degrades_on_402(self):
        cfg = _mini_cfg()
        with patch("src.llm._get_client", return_value=_FakeClient()):
            async def go():
                out = []
                async for delta, usage in stream_chat(cfg, [{"role": "user", "content": "hi"}]):
                    out.append((delta, usage))
                return out

            out = asyncio.run(go())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0].content, "降级通道兜底成功", "402 后应走非流式降级兜底")

    def test_stream_chat_all_fail_raises(self):
        cfg = _mini_cfg()

        class _AllFail:
            class _C:
                async def create(self, **kwargs):
                    raise Simulated402("Insufficient Balance")

            @property
            def chat(self):
                return MagicMock(completions=self._C())

        with patch("src.llm._get_client", return_value=_AllFail()):
            async def go():
                async for _ in stream_chat(cfg, []):
                    pass

            with self.assertRaises(Simulated402):
                asyncio.run(go())

    def test_chat_degrades_to_fallback(self):
        cfg = _mini_cfg()
        ok_resp = MagicMock()
        with patch("src.llm._call_role",
                   new=AsyncMock(side_effect=[Simulated402("402"), ok_resp])):
            async def go():
                return await chat(cfg, [], role="default")

            self.assertIs(asyncio.run(go()), ok_resp, "主角色 402 → fallback 成功")


class TestAgentFriendly(unittest.TestCase):
    """agent 层：双通道全失败 → 返回友好提示文本，不抛异常。"""

    def test_nostream_402_returns_friendly(self):
        a = Agent(stream=False)
        with patch("src.agent.chat", new=AsyncMock(side_effect=Simulated402("Insufficient Balance"))):
            out = asyncio.run(a.run("问题"))
        self.assertTrue(out.startswith("⚠ 模型调用失败"), f"非流式应友好返回，实际：{out[:40]}")
        self.assertIn("Insufficient Balance", out, "提示里带失败原因")
        self.assertEqual(a.tracer.events[-1]["type"], "run_end", "失败也收尾 trace")

    def test_stream_402_returns_friendly(self):
        a = Agent(stream=True)

        async def fake_stream(*args, **kwargs):
            raise Simulated402("Insufficient Balance")
            yield  # pragma: no cover

        with patch("src.agent.stream_chat", new=fake_stream):
            out = asyncio.run(a.run("问题"))
        self.assertTrue(out.startswith("⚠ 模型调用失败"), f"流式应友好返回，实际：{out[:40]}")
        self.assertEqual(a.tracer.events[-1]["type"], "run_end", "失败也收尾 trace")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if r.wasSuccessful() else 1)
