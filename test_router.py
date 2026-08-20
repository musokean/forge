"""路由判断测试：规则预判秒回（问候/简单问答/辩论/多任务/规划）+ 模型兜底 + 超时降级。

运行：python test_router.py
依赖：managed venv（不联网——模型调用全 mock，规则路径完全不调模型）
"""
import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, ".")

from src.router import route, _rule_route, ROUTE_TIMEOUT


class TestRuleRoute(unittest.TestCase):
    """规则预判：命中即返回，绝不调模型。"""

    def test_greeting_single(self):
        for q in ("你好", "嗨", "hello", "在吗", "谢谢", "早上好"):
            self.assertEqual(_rule_route(q)["type"], "single", q)

    def test_simple_question_single(self):
        for q in ("今天深圳天气怎么样", "1+1 等于几", "现在几点", "forge 是什么",
                  "帮我把这段文字翻译成英文", "推荐几本 AI 的书"):
            self.assertEqual(_rule_route(q)["type"], "single", q)

    def test_debate_rules(self):
        for q in ("该不该用 AI 写开发信", "选 A 还是 B", "这个方案值不值得做",
                  "对比一下两个方案的利弊"):
            self.assertEqual(_rule_route(q)["type"], "debate", q)

    def test_plan_rules(self):
        for q in ("调研 A 和 B 的差异并给出选型建议", "梳理项目结构并写份报告",
                  "帮我制定一个推广计划"):
            self.assertEqual(_rule_route(q)["type"], "plan", q)

    def test_parallel_rules(self):
        for q in ("分别查一下深圳和广州的天气", "查天气、查汇率、算个数",
                  "同时算一下 A 和 B 的面积"):
            self.assertEqual(_rule_route(q)["type"], "parallel", q)

    def test_unknown_returns_none(self):
        """规则拿不准 → None，交给模型兜底。"""
        self.assertIsNone(_rule_route("帮我分析一下这个数据文件里的异常值"))

    def test_empty_single(self):
        self.assertEqual(_rule_route("  ")["type"], "single")


class TestModelFallback(unittest.TestCase):
    """规则未命中时走模型兜底。"""

    def test_unknown_question_uses_model(self):
        with patch("src.router._call_role") as mock_call:
            mock_call.return_value = SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"type": "single", "subtasks": [], "question": "q"}'))
            ])
            d = asyncio.run(route("帮我分析一下这个数据文件里的异常值", {"router": {"role": "default"}}))
        self.assertEqual(d["type"], "single")
        mock_call.assert_called_once()

    def test_model_debate(self):
        with patch("src.router._call_role") as mock_call:
            mock_call.return_value = SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"type": "debate", "question": "要不要外包"}'))
            ])
            d = asyncio.run(route("帮我分析一下这个数据文件里的异常值", {"router": {"role": "default"}}))
        self.assertEqual(d["type"], "debate")
        self.assertEqual(d["question"], "要不要外包")

    def test_model_bad_json_falls_back_single(self):
        with patch("src.router._call_role") as mock_call:
            mock_call.return_value = SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content="不是 JSON"))
            ])
            d = asyncio.run(route("帮我分析一下这个数据文件里的异常值", {"router": {"role": "default"}}))
        self.assertEqual(d["type"], "single")

    def test_model_exception_falls_back_single(self):
        with patch("src.router._call_role", side_effect=RuntimeError("boom")):
            d = asyncio.run(route("帮我分析一下这个数据文件里的异常值", {"router": {"role": "default"}}))
        self.assertEqual(d["type"], "single")

    def test_model_timeout_falls_back_single(self):
        """模型挂起超过 ROUTE_TIMEOUT → 降级 single（不再卡死）。"""
        async def slow(*a, **k):
            await asyncio.sleep(999)
        with patch("src.router._call_role", new=slow):
            d = asyncio.run(route("帮我分析一下这个数据文件里的异常值", {"router": {"role": "default"}}))
        self.assertEqual(d["type"], "single")

    def test_timeout_is_short(self):
        """路由判断超时必须短（≤6s，不阻塞用户）。"""
        async def slow(*a, **k):
            await asyncio.sleep(999)
        with patch("src.router._call_role", new=slow):
            import time
            t0 = time.monotonic()
            asyncio.run(route("帮我分析一下这个数据文件里的异常值", {"router": {"role": "default"}}))
            el = time.monotonic() - t0
        self.assertLessEqual(el, 6, f"路由超时兜底耗时 {el:.1f}s，超过 6s 上限")
        self.assertLessEqual(ROUTE_TIMEOUT, 6)


class TestRouteGreetingNoModel(unittest.TestCase):
    """关键回归：问候语绝不调模型（反馈「你好也卡」）。"""

    def test_greeting_skips_model(self):
        with patch("src.router._call_role") as mock_call:
            d = asyncio.run(route("你好", {}))
        self.assertEqual(d["type"], "single")
        mock_call.assert_not_called()  # 规则命中，模型零调用

    def test_common_questions_skip_model(self):
        with patch("src.router._call_role") as mock_call:
            d = asyncio.run(route("今天天气怎么样", {}))
        self.assertEqual(d["type"], "single")
        mock_call.assert_not_called()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
