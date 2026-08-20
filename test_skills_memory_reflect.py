"""四新模块综合测试：Skill 技能包 / Memory 长期记忆 / Reflect 反思 / Supervisor 规划执行。

运行：python test_skills_memory_reflect.py
依赖：managed venv（src 可 import，不联网——模型调用全部 mock）
"""
import asyncio
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, ".")

from src.skills import (
    SKILLS, activate, deactivate, list_skills, active_skills,
    compose_prompt, schema_filter, skill_status_line,
)
from src.memory import MemoryStore
from src.structured import RoleBrief


# ===================== 1) Skill 技能包 =====================
class TestSkills(unittest.TestCase):
    def setUp(self):
        from src import skills
        skills._active = set()

    def test_registry_has_builtin(self):
        for n in ("coding", "writing", "research", "knowledge"):
            self.assertIn(n, SKILLS, f"缺内置技能 {n}")

    def test_activate_and_prompt(self):
        activate("coding")
        self.assertIn("编程助手", compose_prompt("BASE"))
        self.assertTrue(active_skills())

    def test_deactivate(self):
        activate("research")
        deactivate("research")
        self.assertNotIn("research", active_skills())
        self.assertEqual(compose_prompt("BASE"), "BASE")

    def test_unknown_skill_raises(self):
        with self.assertRaises(ValueError):
            activate("no_such_skill")

    def test_schema_filter_whitelist(self):
        activate("coding")  # 白名单：文件/命令类
        full = [{"function": {"name": n}} for n in
                ("read_file", "kb_search", "web_search", "run_command")]
        kept = {s["function"]["name"] for s in schema_filter(full)}
        self.assertIn("read_file", kept)
        self.assertIn("run_command", kept)
        self.assertNotIn("kb_search", kept)  # 知识库工具不在 coding 白名单
        self.assertNotIn("web_search", kept)

    def test_schema_filter_no_active_returns_full(self):
        full = [{"function": {"name": "read_file"}}]
        self.assertEqual(schema_filter(full), full)

    def test_status_line(self):
        self.assertEqual(skill_status_line(), "（无）")
        activate("writing")
        self.assertIn("writing", skill_status_line())


# ===================== 2) Memory 长期记忆 =====================
class TestMemory(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "m.db")
        self.m = MemoryStore(self.path)

    def test_remember_and_duplicate(self):
        ok, _ = self.m.remember("小明做跨境电商，美容仪器方向")
        self.assertTrue(ok)
        ok2, msg2 = self.m.remember("小明做跨境电商，美容仪器方向")
        self.assertFalse(ok2)
        self.assertIn("去重", msg2)
        self.assertEqual(self.m.stats()["count"], 1)

    def test_auto_remember_patterns(self):
        got = self.m.auto_remember("我喜欢简洁的回答，不要啰嗦")
        self.assertIsNotNone(got)
        self.assertIn("简洁", got)
        self.assertIsNone(self.m.auto_remember("今天天气怎么样"))  # 不命中模式

    def test_recall_and_hit_count(self):
        self.m.remember("小明负责 MYCHWAY 的产品知识库")
        hits = self.m.recall("知识库")
        self.assertEqual(len(hits), 1)
        hits2 = self.m.recall("知识库")
        self.assertEqual(hits2[0]["hit_count"], 2)  # 命中计数递增

    def test_forget_and_clear(self):
        self.m.remember("甲喜欢喝咖啡")
        self.m.remember("乙喜欢喝茶")
        self.assertEqual(self.m.forget("咖啡"), 1)
        self.assertEqual(self.m.stats()["count"], 1)
        self.assertEqual(self.m.clear(), 1)
        self.assertEqual(self.m.stats()["count"], 0)

    def test_compose_context(self):
        self.m.remember("小明是做跨境电商的")
        ctx = self.m.compose_context("跨境电商")
        self.assertIn("长期记忆", ctx)
        self.assertIn("跨境电商", ctx)
        self.assertEqual(self.m.compose_context("无关词xyz"), "")


# ===================== 3) Reflect 反思自纠错 =====================
def _fake_resp(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class TestReflect(unittest.TestCase):
    def test_evaluate_parses_score(self):
        from src.reflect import evaluate_answer
        cfg = {"reflect": {"enabled": True}}
        with patch("src.reflect.chat", AsyncMock(return_value=_fake_resp(
                '{"score": 4, "issues": ["漏了关键结论"], "suggestion": "补充结论"}'))):
            v = asyncio.run(evaluate_answer(cfg, "q", "a", judge_role="fallback"))
        self.assertEqual(v["score"], 4)
        self.assertEqual(v["suggestion"], "补充结论")

    def test_evaluate_failure_returns_none(self):
        from src.reflect import evaluate_answer
        with patch("src.reflect.chat", AsyncMock(side_effect=Exception("boom"))):
            v = asyncio.run(evaluate_answer({}, "q", "a"))
        self.assertIsNone(v)  # 评审失败静默，不阻塞

    def test_evaluate_invalid_json_returns_none(self):
        from src.reflect import evaluate_answer
        with patch("src.reflect.chat", AsyncMock(return_value=_fake_resp("不是JSON"))):
            v = asyncio.run(evaluate_answer({}, "q", "a"))
        self.assertIsNone(v)

    def test_refine_failure_keeps_original(self):
        from src.reflect import refine_answer
        with patch("src.reflect.chat", AsyncMock(side_effect=Exception("boom"))):
            out = asyncio.run(refine_answer({}, "q", "原答案", "意见"))
        self.assertEqual(out, "原答案")  # 修正失败绝不比不纠更差

    def test_maybe_reflect_disabled_returns_unchanged(self):
        from src.agent import Agent
        with patch("src.agent.load_config", return_value={"reflect": {"enabled": False}}), \
             patch("src.agent.compose_prompt", side_effect=lambda s: s), \
             patch("src.agent.schema_filter", side_effect=lambda s: s):
            agent = Agent.__new__(Agent)
            agent.cfg = {"reflect": {"enabled": False}}
            out = asyncio.run(agent._maybe_reflect("q", "答案"))
        self.assertEqual(out, "答案")

    def test_maybe_reflect_low_score_refines(self):
        from src.agent import Agent
        from src import reflect as reflect_mod
        agent = Agent.__new__(Agent)
        agent.cfg = {"reflect": {"enabled": True, "min_score": 6, "max_rounds": 1, "judge_role": "fallback"}}
        agent.stream = False
        agent.role = "default"
        with patch.object(reflect_mod, "evaluate_answer", AsyncMock(return_value={
                "score": 3, "issues": ["太短"], "suggestion": "展开讲"})), \
             patch.object(reflect_mod, "refine_answer", AsyncMock(return_value="修正后的完整答案")):
            out = asyncio.run(agent._maybe_reflect("q", "原答案"))
        self.assertEqual(out, "修正后的完整答案")

    def test_maybe_reflect_good_score_unchanged(self):
        from src.agent import Agent
        from src import reflect as reflect_mod
        agent = Agent.__new__(Agent)
        agent.cfg = {"reflect": {"enabled": True, "min_score": 6, "max_rounds": 1}}
        agent.stream = False
        agent.role = "default"
        with patch.object(reflect_mod, "evaluate_answer", AsyncMock(return_value={
                "score": 9, "issues": [], "suggestion": ""})):
            out = asyncio.run(agent._maybe_reflect("q", "好答案"))
        self.assertEqual(out, "好答案")


# ===================== 4) Supervisor 规划执行 =====================
class TestSupervisor(unittest.TestCase):
    def test_plan_schema(self):
        from src.orchestrator import Plan, PlanItem
        p = Plan(plan_title="调研", subtasks=[
            PlanItem(title="查 A", instruction="调研 A 的特点"),
            PlanItem(title="查 B", instruction="调研 B 的特点"),
        ])
        self.assertEqual(len(p.subtasks), 2)
        self.assertEqual(p.subtasks[1].title, "查 B")

    def test_supervisor_normal_flow(self):
        from src.orchestrator import run_supervised, Plan, PlanItem
        plan = Plan(plan_title="调研", subtasks=[
            PlanItem(title="A", instruction="调研 A"),
            PlanItem(title="B", instruction="调研 B"),
        ])
        briefs = [
            RoleBrief(name="A", stance="A 便宜", key_points=["点1"]),
            RoleBrief(name="B", stance="B 快", key_points=["点2"]),
        ]
        with patch("src.orchestrator.load_config", return_value={}), \
             patch("src.orchestrator.ask_structured", AsyncMock(return_value=(plan, None))), \
             patch("src.orchestrator.run_parallel", AsyncMock(return_value=briefs)), \
             patch("src.orchestrator.chat", AsyncMock(return_value=_fake_resp("合并后的最终答案"))):
            out = asyncio.run(run_supervised("调研 A 和 B"))
        self.assertEqual(out, "合并后的最终答案")

    def test_supervisor_plan_fail_degrades_to_direct(self):
        from src.orchestrator import run_supervised
        from src.structured import StructuredError
        with patch("src.orchestrator.load_config", return_value={}), \
             patch("src.orchestrator.ask_structured", AsyncMock(side_effect=StructuredError("x"))), \
             patch("src.agent.Agent.run", AsyncMock(return_value="直答结果")):
            out = asyncio.run(run_supervised("复杂任务"))
        self.assertEqual(out, "直答结果")  # 拆解失败自动降级，不崩

    def test_supervisor_failed_subtask_still_merges(self):
        from src.orchestrator import run_supervised, Plan, PlanItem
        plan = Plan(plan_title="t", subtasks=[PlanItem(title="A", instruction="a")])
        briefs = ["子任务执行失败"]  # 非 RoleBrief → 视为失败
        with patch("src.orchestrator.load_config", return_value={}), \
             patch("src.orchestrator.ask_structured", AsyncMock(return_value=(plan, None))), \
             patch("src.orchestrator.run_parallel", AsyncMock(return_value=briefs)), \
             patch("src.orchestrator.chat", AsyncMock(return_value=_fake_resp("部分失败但合并成功"))):
            out = asyncio.run(run_supervised("任务"))
        self.assertEqual(out, "部分失败但合并成功")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
