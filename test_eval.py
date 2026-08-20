"""#13 评估模块测试：黄金集加载/保存/新增 + 关键词命中 + judge 打分 + 双通道判定 + 报告导出。

运行：python test_eval.py
依赖：managed venv（不联网——模型调用全部 mock）
"""
import asyncio
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, ".")

from src.eval import Evaluator, EvalResult, _keyword_hit, DEFAULT_GOLDEN, EVAL_EXPORT_DIR


class TestKeywordHit(unittest.TestCase):
    def test_case_insensitive(self):
        self.assertTrue(_keyword_hit("Forge is an AI assistant", "forge"))
        self.assertTrue(_keyword_hit("答案是 391", "391"))

    def test_no_hit(self):
        self.assertFalse(_keyword_hit("答案是 392", "391"))
        self.assertFalse(_keyword_hit("", "x"))

    def test_chinese_substring(self):
        self.assertTrue(_keyword_hit("forge 是自动任务调度器", "自动任务"))
        self.assertFalse(_keyword_hit("forge 是任务调度器", "自动任务"))


class TestGoldenManagement(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "golden.yaml")
        self.ev = Evaluator(self.path)

    def test_first_load_writes_default(self):
        cases = self.ev.load_golden()
        self.assertEqual(len(cases), len(DEFAULT_GOLDEN))
        self.assertTrue(os.path.exists(self.path))

    def test_roundtrip_preserves_fields(self):
        self.ev.load_golden()
        cases = self.ev.load_golden()
        self.assertEqual(cases[0]["task"], DEFAULT_GOLDEN[0]["task"])
        self.assertEqual(cases[0]["keywords"], ["391"])
        self.assertEqual(cases[0]["min_score"], 6)

    def test_add_case(self):
        self.ev.load_golden()
        ok, msg = self.ev.add_case("1+1 等于几", ["2"], 7, "加法")
        self.assertTrue(ok)
        self.assertIn("1+1", msg)
        cases = self.ev.load_golden()
        self.assertEqual(len(cases), len(DEFAULT_GOLDEN) + 1)
        self.assertEqual(cases[-1]["keywords"], ["2"])
        self.assertEqual(cases[-1]["min_score"], 7)

    def test_add_empty_rejected(self):
        self.ev.load_golden()
        ok, _ = self.ev.add_case("  ", [], 6)
        self.assertFalse(ok)


def _fake_resp(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class TestRunCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ev = Evaluator(os.path.join(self.dir, "golden.yaml"))

    def test_keyword_pass_judge_fail(self):
        """关键词全命中但 judge 低分 → 整体不过（双通道都要过）。"""
        case = {"task": "1+1=?", "keywords": ["2"], "min_score": 6}
        fake_agent = SimpleNamespace(
            run=AsyncMock(return_value="答案是 2"),
            total_tokens={"prompt": 10, "completion": 5},
        )
        with patch("src.eval.load_config", return_value={}), \
             patch("src.eval.evaluate_answer", AsyncMock(return_value={
                 "score": 3, "issues": ["太简略"], "suggestion": "展开"})):
            r = asyncio.run(self.ev.run_case(case, agent=fake_agent))
        self.assertTrue(r.keyword_pass)
        self.assertFalse(r.judge_pass)
        self.assertFalse(r.passed)
        self.assertEqual(r.judge_score, 3)

    def test_keyword_fail_judge_pass(self):
        """关键词未全命中 → 程序化硬指标不过，judge 高分也救不回来。"""
        case = {"task": "q", "keywords": ["x", "y"], "min_score": 6}
        fake_agent = SimpleNamespace(
            run=AsyncMock(return_value="只有 x"),
            total_tokens={"prompt": 1, "completion": 1},
        )
        with patch("src.eval.load_config", return_value={}), \
             patch("src.eval.evaluate_answer", AsyncMock(return_value={
                 "score": 9, "issues": [], "suggestion": ""})):
            r = asyncio.run(self.ev.run_case(case, agent=fake_agent))
        self.assertFalse(r.keyword_pass)
        self.assertFalse(r.passed)

    def test_all_pass(self):
        case = {"task": "q", "keywords": ["forge"], "min_score": 6}
        fake_agent = SimpleNamespace(
            run=AsyncMock(return_value="forge 是 AI 助手"),
            total_tokens={"prompt": 1, "completion": 1},
        )
        with patch("src.eval.load_config", return_value={}), \
             patch("src.eval.evaluate_answer", AsyncMock(return_value={
                 "score": 8, "issues": [], "suggestion": ""})):
            r = asyncio.run(self.ev.run_case(case, agent=fake_agent))
        self.assertTrue(r.passed)

    def test_judge_unavailable_only_keyword(self):
        """评审不可用（None）→ 不扣分，只看关键词通道（断网可跑回归）。"""
        case = {"task": "q", "keywords": ["391"], "min_score": 6}
        fake_agent = SimpleNamespace(
            run=AsyncMock(return_value="结果是 391"),
            total_tokens={"prompt": 1, "completion": 1},
        )
        with patch("src.eval.load_config", return_value={}), \
             patch("src.eval.evaluate_answer", AsyncMock(return_value=None)):
            r = asyncio.run(self.ev.run_case(case, agent=fake_agent))
        self.assertIsNone(r.judge_score)
        self.assertTrue(r.judge_pass)
        self.assertTrue(r.passed)

    def test_judge_raises_still_runs(self):
        """评审链路抛异常 → 不阻塞回归，结果交给关键词通道。"""
        case = {"task": "q", "keywords": ["ok"], "min_score": 6}
        fake_agent = SimpleNamespace(
            run=AsyncMock(return_value="ok fine"),
            total_tokens={"prompt": 1, "completion": 1},
        )
        with patch("src.eval.load_config", return_value={}), \
             patch("src.eval.evaluate_answer", AsyncMock(side_effect=Exception("boom"))):
            r = asyncio.run(self.ev.run_case(case, agent=fake_agent))
        self.assertIsNone(r.judge_score)
        self.assertTrue(r.passed)

    def test_agent_default_created(self):
        """不传 agent 时自动新建 Agent（非流式 + 写操作放行 + 无 spinner）。"""
        from src.approval import Approver
        case = {"task": "q", "keywords": [], "min_score": 6}
        with patch("src.eval.load_config", return_value={}), \
             patch("src.eval.evaluate_answer", AsyncMock(return_value=None)):
            inst = self.ev._new_agent()
            self.assertFalse(inst.stream)
            self.assertFalse(inst.show_spinner)
            self.assertIsInstance(inst.approver, Approver)


class TestRunAllAndReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ev = Evaluator(os.path.join(self.dir, "golden.yaml"))

    def test_run_all_parallel(self):
        cases = [
            {"task": f"t{i}", "keywords": [f"k{i}"], "min_score": 6} for i in range(4)
        ]
        fake = SimpleNamespace(
            run=AsyncMock(side_effect=["k0 hit", "k1 hit", "k2 hit", "k3 hit"]),
            total_tokens={"prompt": 1, "completion": 1},
        )
        with patch("src.eval.load_config", return_value={}), \
             patch("src.eval.evaluate_answer", AsyncMock(return_value={"score": 8, "issues": [], "suggestion": ""})):
            results = asyncio.run(self.ev.run_all(cases, agent_factory=lambda: fake))
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.passed for r in results))

    def test_report_summary(self):
        results = [
            EvalResult(task="A", answer="ok", keywords=["ok"], hits=[True], min_score=6,
                       judge_score=8, passed=True, ms=100),
            EvalResult(task="B", answer="bad", keywords=["x"], hits=[False], min_score=6,
                       judge_score=3, passed=False, ms=200),
        ]
        rep = self.ev.report(results)
        self.assertIn("1/2 通过", rep)
        self.assertIn("✅", rep)
        self.assertIn("❌", rep)

    def test_export_markdown(self):
        results = [
            EvalResult(task="A", answer="ok", keywords=["ok"], hits=[True], min_score=6,
                       judge_score=8, passed=True, ms=100),
            EvalResult(task="B", answer="bad answer here", keywords=["x"], hits=[False], min_score=6,
                       judge_score=3, passed=False, ms=200),
        ]
        path = self.ev.export_markdown(results, name="test-eval.md")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("1/2 通过", content)
        self.assertIn("A", content)
        self.assertIn("未通过详情", content)
        self.assertIn("bad answer here", content)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
