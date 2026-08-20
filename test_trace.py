"""M3 #7 trace 基础测试：每角色/每步可观测（A09 补全）。

框架无关（unittest.mock），裸跑 python test_trace.py 即可。
"""
import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, ".")

from src.agent import Agent
from src.trace import Tracer


def _resp(content=None, tool_calls=None, p=10, c=5):
    resp = MagicMock()
    resp.usage.prompt_tokens = p
    resp.usage.completion_tokens = c
    msg = resp.choices[0].message
    msg.content = content
    msg.tool_calls = tool_calls
    return resp


def _tc(name, args, id_="call_1"):
    tc = MagicMock()
    tc.id = id_
    tc.function.name = name
    tc.function.arguments = args
    return tc


def _delta(content=None, reasoning=None, tcs=None):
    d = MagicMock()
    d.content = content
    d.reasoning_content = reasoning
    d.tool_calls = tcs
    return d


async def _gen_tool():
    yield _delta(tcs=[_tc("calculator", '{"expression":"2*3"}', id_="c1")]), None


async def _gen_answer():
    yield _delta(content="6"), None


class TestTracerBasics(unittest.TestCase):
    def test_record_and_summary(self):
        tr = Tracer("正方", "default", "qwen3")
        tr.record("run_start", task="算 1+1")
        tr.record("llm_step", step=0, tokens=15, ms=100)
        tr.record("tool", name="calculator", arg_summary="{'expr':'1+1'}", result_summary="2", ms=5)
        tr.record("run_end", ms=200, prompt_tokens=10, completion_tokens=5)
        s = tr.summary()
        self.assertEqual(s["agent"], "正方")
        self.assertEqual(s["model"], "qwen3")
        self.assertEqual(s["steps"], 1)
        self.assertIn("calculator", s["tools"][0])
        self.assertEqual(s["llm_ms"], 100)
        self.assertEqual(s["tool_ms"], 5)
        self.assertEqual(s["total_ms"], 200)
        self.assertEqual(s["prompt_tokens"], 10)

    def test_jsonl_roundtrip(self):
        tr = Tracer("forge")
        tr.record("run_start", task="x")
        tr.record("run_end", ms=1)
        path = os.path.join(tempfile.mkdtemp(prefix="forge_trace_"), "t.jsonl")
        tr.to_jsonl(path)
        ev = Tracer.load_jsonl(path)
        self.assertEqual(len(ev), 2)
        self.assertEqual(ev[0]["type"], "run_start")
        self.assertEqual(ev[0]["agent"], "forge")  # 落盘带 agent 名


class TestAgentTrace(unittest.TestCase):
    def test_nostream_full_flow(self):
        a = Agent(stream=False)
        r1 = _resp(tool_calls=[_tc("calculator", '{"expression":"1+1"}')])
        r2 = _resp(content="结果是 2")
        with patch("src.agent.chat", new=AsyncMock(side_effect=[r1, r2])):
            out = asyncio.run(a.run("算 1+1"))
        self.assertEqual(out, "结果是 2")
        types = [e["type"] for e in a.tracer.events]
        self.assertEqual(types, ["run_start", "llm_step", "tool", "llm_step", "answer", "run_end"],
                         "非流式全流程事件序列应完整")
        tool = [e for e in a.tracer.events if e["type"] == "tool"][0]
        self.assertEqual(tool["name"], "calculator")
        self.assertIn("1+1", tool["arg_summary"])
        self.assertIn("2", tool["result_summary"])
        end = [e for e in a.tracer.events if e["type"] == "run_end"][0]
        self.assertGreaterEqual(end["prompt_tokens"] + end["completion_tokens"], 0)
        self.assertIsNotNone(end["ms"])

    def test_stream_full_flow(self):
        a = Agent(stream=True)
        calls = {"n": 0}

        async def fake_stream(*args, **kwargs):
            calls["n"] += 1
            gen = _gen_tool() if calls["n"] == 1 else _gen_answer()
            async for x in gen:
                yield x

        with patch("src.agent.stream_chat", new=fake_stream):
            out = asyncio.run(a.run("算 2*3"))
        self.assertEqual(out, "6")
        types = [e["type"] for e in a.tracer.events]
        self.assertEqual(types, ["run_start", "llm_step", "tool", "llm_step", "answer", "run_end"],
                         "流式全流程事件序列应完整")
        tool = [e for e in a.tracer.events if e["type"] == "tool"][0]
        self.assertEqual(tool["name"], "calculator")
        self.assertIn("2*3", tool["arg_summary"])

    def test_multi_agent_independent(self):
        a1 = Agent(stream=False, name="正方")
        a2 = Agent(stream=False, name="反方")
        with patch("src.agent.chat", new=AsyncMock(return_value=_resp(content="观点"))):
            asyncio.run(a1.run("论题"))
            asyncio.run(a2.run("论题"))
        self.assertIsNot(a1.tracer, a2.tracer, "多智能体 tracer 各自独立")
        self.assertEqual(a1.tracer.agent, "正方")
        self.assertEqual(a2.tracer.agent, "反方")
        for a in (a1, a2):
            types = [e["type"] for e in a.tracer.events]
            self.assertEqual(types, ["run_start", "llm_step", "answer", "run_end"])

    def test_run_structured_traced(self):
        a = Agent(stream=False)
        obj = MagicMock()
        obj.model_dump.return_value = {"name": "x", "stance": "支持"}
        with patch("src.agent.ask_structured", new=AsyncMock(return_value=(obj, _resp(content="{}")))):
            out = asyncio.run(a.run_structured("任务", object))
        self.assertIsNotNone(out)
        types = [e["type"] for e in a.tracer.events]
        self.assertEqual(types, ["run_start", "answer", "run_end"])
        ans = [e for e in a.tracer.events if e["type"] == "answer"][0]
        self.assertIn("stance", ans["content"])

    def test_forced_answer_records_run_end(self):
        # 连续 3 次相同工具调用 → 强制回答收敛（防死循环路径），run_end 必须收尾
        a = Agent(stream=False)
        r = _resp(tool_calls=[_tc("calculator", '{"expression":"1+1"}')])
        with patch("src.agent.chat", new=AsyncMock(return_value=r)):
            out = asyncio.run(a.run("任务"))
        self.assertEqual(out, "", "强制回答后模型仍回 tool_calls → 兜底返回 content")
        types = [e["type"] for e in a.tracer.events]
        self.assertEqual(types[-1], "run_end", "强制收敛出口也要收尾 trace")
        self.assertEqual(types.count("tool"), 2, "第 3 次相同调用被拦截（不执行）")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if r.wasSuccessful() else 1)
