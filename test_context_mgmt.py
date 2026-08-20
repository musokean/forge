"""M3 #3 上下文管理补全测试：滚动摘要 + 工具输出裁剪（A05b 杠杆②）。

框架无关（unittest.mock），裸跑 python test_context_mgmt.py 即可。
"""
import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, ".")

from src.agent import Agent


def _mk_agent(max_context_tokens=8000, max_tool_output=1500):
    return Agent(stream=False, max_context_tokens=max_context_tokens, max_tool_output=max_tool_output)


def _mk_history(rounds, size=1500):
    """构造 rounds 轮历史，每轮 user + assistant 两条 size 字符消息。"""
    msgs = [{"role": "system", "content": "你是 forge"}]
    for i in range(rounds):
        msgs.append({"role": "user", "content": f"Q{i} " + "x" * (size - 2)})
        msgs.append({"role": "assistant", "content": f"A{i} " + "y" * (size - 2)})
    return msgs


class TestClipToolOutput(unittest.TestCase):
    def test_short_output_unchanged(self):
        a = _mk_agent()
        text = "小输出"
        out = asyncio.run(a._clip_tool_output(text))
        self.assertEqual(out, text, "小输出应原样返回、不触发摘要")

    def test_long_output_summarized(self):
        a = _mk_agent()
        big = "z" * 5000
        with patch.object(a, "_summarize", new=AsyncMock(return_value="关键数据：42")):
            out = asyncio.run(a._clip_tool_output(big))
        self.assertIn("已压缩 5000 字符", out)
        self.assertIn("关键数据：42", out)
        self.assertLess(len(out), 200)

    def test_long_output_truncate_on_summary_fail(self):
        a = _mk_agent(max_tool_output=100)
        big = "z" * 5000
        with patch.object(a, "_summarize", new=AsyncMock(return_value=None)):
            out = asyncio.run(a._clip_tool_output(big))
        self.assertIn("[已截断", out)
        self.assertLessEqual(len(out), 100 + 40)


class TestSummarize(unittest.TestCase):
    def test_summarize_success(self):
        a = _mk_agent()
        resp = MagicMock()
        resp.choices[0].message.content = " 摘要正文 "
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        with patch("src.agent.chat", new=AsyncMock(return_value=resp)) as m_chat:
            out = asyncio.run(a._summarize("很长很长" * 100, "压缩测试"))
        self.assertEqual(out, "摘要正文")
        # 摘要调用记入 total_tokens（诚实记账）
        self.assertEqual(a.total_tokens["prompt"] + a.total_tokens["completion"], 15)

    def test_summarize_failure_returns_none(self):
        a = _mk_agent()
        with patch("src.agent.chat", new=AsyncMock(side_effect=RuntimeError("上游挂了"))):
            out = asyncio.run(a._summarize("很长很长" * 100, "压缩测试"))
        self.assertIsNone(out, "摘要失败返回 None，不抛异常")

    def test_summarize_empty(self):
        a = _mk_agent()
        self.assertEqual(asyncio.run(a._summarize("", "压缩测试")), "")


class TestRollSummary(unittest.TestCase):
    def test_rolls_oldest_round_when_over_threshold(self):
        a = _mk_agent(max_context_tokens=8000)  # 阈值 6800
        a.messages = _mk_history(6, size=1500)  # ~6*(1500+1500)=18000 > 6800
        with patch.object(a, "_summarize", new=AsyncMock(return_value="背景：用户问了 6 个问题")):
            asyncio.run(a._maybe_roll_summary())
        texts = [m["content"] for m in a.messages]
        self.assertIn("[早期对话摘要，作为背景参考，不需要回应]", texts, "最早一轮应被摘要对替换")
        self.assertNotIn("Q0 ", texts, "最早的原始消息应被替换")
        self.assertLess(a._context_tokens(), 8000, "摘要后预算内")
        # 摘要以 user/assistant 对存在（不破坏交替结构）
        roles = [m["role"] for m in a.messages]
        self.assertEqual(roles[:3], ["system", "user", "assistant"])

    def test_no_summary_when_within_budget(self):
        a = _mk_agent(max_context_tokens=8000)
        a.messages = _mk_history(2, size=1500)  # ~6000 < 6800 不触发
        with patch.object(a, "_summarize", new=AsyncMock(return_value="不应调用")) as m:
            asyncio.run(a._maybe_roll_summary())
        m.assert_not_awaited()
        self.assertEqual(len(a.messages), 5, "预算内历史不动")

    def test_summary_fail_falls_back_to_drop(self):
        a = _mk_agent(max_context_tokens=8000)
        a.messages = _mk_history(6, size=1500)
        with patch.object(a, "_summarize", new=AsyncMock(return_value=None)):
            asyncio.run(a._maybe_roll_summary())  # 不应抛异常
        self.assertNotIn("Q0 ", [m["content"] for m in a.messages], "摘要失败也压缩了最早轮")

    def test_only_one_round_left_falls_back_to_hard_trim(self):
        a = _mk_agent(max_context_tokens=200)
        a.messages = _mk_history(3, size=500)  # 超预算但只有 3 轮
        before = a._context_tokens()
        with patch.object(a, "_summarize", new=AsyncMock(return_value="摘要")):
            asyncio.run(a._maybe_roll_summary())  # 不应崩、不应死循环
        self.assertLess(a._context_tokens(), before, "滚摘要后 token 应显著下降")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if r.wasSuccessful() else 1)
