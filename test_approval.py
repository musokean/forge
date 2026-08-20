"""M3 #4 审批层测试：写操作执行前的人工/可编程确认（A06 审批补全）。

框架无关（unittest.mock），裸跑 python test_approval.py 即可。
"""
import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, ".")

from src.agent import Agent
from src.approval import Approver


def _resp(content=None, tool_calls=None):
    resp = MagicMock()
    resp.usage.prompt_tokens = 1
    resp.usage.completion_tokens = 1
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


def _p(path):
    return path.replace(os.sep, "/")  # Windows 反斜杠在 mock JSON 里需转正斜杠


class TestApprover(unittest.TestCase):
    def test_auto_approve(self):
        a = Approver(auto_approve=True)
        self.assertTrue(a.approve("write_file", "写文件"))
        self.assertTrue(a.approve("run_command", "rm -rf /"))
        self.assertEqual(len(a.decisions), 2)

    def test_auto_reject(self):
        a = Approver(mode="auto_reject")
        self.assertFalse(a.approve("write_file", "x"))
        self.assertEqual(a.decisions[0][2], False)

    def test_callback(self):
        calls = []
        a = Approver(callback=lambda name, summary: calls.append((name, summary)) or (name == "write_file"))
        self.assertTrue(a.approve("write_file", "ok 写"))
        self.assertFalse(a.approve("edit_file", "no 改"))
        self.assertEqual(calls[0][0], "write_file")
        self.assertIn("ok 写", calls[0][1])

    def test_interactive_yes(self):
        with patch("builtins.input", return_value="y"):
            self.assertTrue(Approver().approve("write_file", "x"))

    def test_interactive_no_and_default(self):
        with patch("builtins.input", return_value="n"):
            self.assertFalse(Approver().approve("write_file", "x"))
        with patch("builtins.input", return_value=""):
            self.assertFalse(Approver().approve("write_file", "x"), "回车默认拒绝")


class TestAgentApproval(unittest.TestCase):
    def test_write_rejected_then_allowed(self):
        tmp = tempfile.mkdtemp(prefix="forge_appr_")
        p1, p2 = os.path.join(tmp, "a.txt"), os.path.join(tmp, "b.txt")
        decisions = []

        def cb(name, summary):
            decisions.append(name)
            return len(decisions) == 2  # 第一次拒、第二次放行

        a = Agent(stream=False, approver=Approver(callback=cb))
        r1 = _resp(tool_calls=[_tc("write_file", f'{{"path": "{_p(p1)}", "content": "x"}}', id_="c1")])
        r2 = _resp(tool_calls=[_tc("write_file", f'{{"path": "{_p(p2)}", "content": "y"}}', id_="c2")])
        r3 = _resp(content="搞定")
        with patch("src.agent.chat", new=AsyncMock(side_effect=[r1, r2, r3])):
            out = asyncio.run(a.run("写两个文件"))
        self.assertEqual(out, "搞定")
        # 第一次被拒：文件不存在，tool 结果带拒绝信息
        self.assertFalse(os.path.exists(p1), "被拒的写操作不应落盘")
        self.assertTrue(os.path.exists(p2), "放行的写操作应落盘")
        tools = [m for m in a.messages if m["role"] == "tool"]
        self.assertIn("拒绝", tools[0]["content"], "拒绝信息喂回模型")
        self.assertEqual(len(decisions), 2)

    def test_readonly_tool_skips_approval(self):
        called = []
        a = Agent(stream=False, approver=Approver(callback=lambda n, s: called.append(n) or True))
        r1 = _resp(tool_calls=[_tc("calculator", '{"expression":"1+1"}')])
        r2 = _resp(content="2")
        with patch("src.agent.chat", new=AsyncMock(side_effect=[r1, r2])):
            asyncio.run(a.run("算数"))
        self.assertEqual(called, [], "只读工具不触发审批")

    def test_default_is_interactive(self):
        # 默认 Agent 不带 approver 时是交互模式（CLI 主对话用）
        self.assertIsInstance(Agent(stream=False).approver, Approver)
        with patch("builtins.input", return_value="y"):
            self.assertTrue(Agent(stream=False).approver.approve("write_file", "x"))

    def test_auto_approve_end_to_end(self):
        # 无人值守（并行/辩论场景）：写操作直接放行，文件真写
        tmp = tempfile.mkdtemp(prefix="forge_appr2_")
        p = os.path.join(tmp, "ok.txt")
        a = Agent(stream=False, approver=Approver(auto_approve=True))
        r1 = _resp(tool_calls=[_tc("write_file", f'{{"path": "{_p(p)}", "content": "hello"}}')])
        r2 = _resp(content="写好了")
        with patch("src.agent.chat", new=AsyncMock(side_effect=[r1, r2])):
            asyncio.run(a.run("写文件"))
        self.assertTrue(os.path.exists(p))
        self.assertEqual(open(p, encoding="utf-8").read(), "hello")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if r.wasSuccessful() else 1)
