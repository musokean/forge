"""自动任务调度器测试（#18）：调度解析 / 增删改查 / 执行落库 / 后台触发 / 启动补跑。

全程 mock 模型（不联网），验证调度器自身逻辑。
"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from src.tasks import (
    TaskScheduler,
    parse_schedule,
    compute_next,
    ScheduleError,
)


# ---------- mock 模型 ----------
class _FakeMsg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = None  # Agent._add_usage 会读取 resp.usage


async def _fake_chat(cfg, messages, role="default", tools=None, retries=3):
    last = messages[-1]["content"] if messages else ""
    return _FakeResp(f"[自动任务产出] {last[:30]}")


class _FakeKB:
    def __init__(self):
        self.calls = []

    def add(self, title, content, kind="inline"):
        self.calls.append((title, content))
        return True, f"已写入 {title}"


class TestParseSchedule(unittest.TestCase):
    def test_interval_cn(self):
        self.assertEqual(parse_schedule("每2小时")[0], "interval")
        self.assertEqual(parse_schedule("每30分钟")[1], "1800")
        self.assertEqual(parse_schedule("每1天")[1], "86400")

    def test_interval_en(self):
        self.assertEqual(parse_schedule("every 2 hours")[1], "7200")
        self.assertEqual(parse_schedule("every 1 day")[1], "86400")

    def test_daily(self):
        st, expr, _ = parse_schedule("每天09:00")
        self.assertEqual(st, "daily")
        self.assertEqual(expr, "09:00")

    def test_once(self):
        st, expr, _ = parse_schedule("once 2026-08-20T14:00")
        self.assertEqual(st, "once")
        self.assertIn("2026-08-20", expr)

    def test_invalid(self):
        with self.assertRaises(ScheduleError):
            parse_schedule("乱写的")


class TestComputeNext(unittest.TestCase):
    from datetime import datetime

    def test_interval(self):
        base = self.datetime(2026, 1, 1, 10, 0, 0)
        nxt = compute_next("interval", "7200", base)
        self.assertEqual(nxt, self.datetime(2026, 1, 1, 12, 0, 0))

    def test_daily_tomorrow(self):
        base = self.datetime(2026, 1, 1, 10, 0, 0)
        nxt = compute_next("daily", "09:00", base)
        self.assertEqual(nxt, self.datetime(2026, 1, 2, 9, 0, 0))

    def test_daily_today(self):
        base = self.datetime(2026, 1, 1, 8, 0, 0)
        nxt = compute_next("daily", "09:00", base)
        self.assertEqual(nxt, self.datetime(2026, 1, 1, 9, 0, 0))


class TestSchedulerCRUD(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "tasks.db")
        self.s = TaskScheduler(self.db)

    def test_add_list_delete(self):
        msg = self.s.add("t1", "每1小时", "做点事")
        self.assertTrue(msg.startswith("✅"))
        self.assertEqual(len(self.s.list_tasks()), 1)
        self.s.delete("t1")
        self.assertEqual(len(self.s.list_tasks()), 0)

    def test_add_invalid_schedule(self):
        msg = self.s.add("t2", "bad", "x")
        self.assertTrue(msg.startswith("⚠"))

    def test_add_empty_prompt(self):
        msg = self.s.add("t3", "每1小时", "")
        self.assertTrue(msg.startswith("⚠"))

    def test_enable_disable(self):
        self.s.add("t4", "每1小时", "x")
        self.s.set_enabled("t4", False)
        self.assertFalse(self.s.get("t4")["enabled"])
        self.s.set_enabled("t4", True)
        self.assertTrue(self.s.get("t4")["enabled"])

    def test_reject_unknown(self):
        self.assertTrue(self.s.delete("nope").startswith("⚠"))


class TestExecute(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "tasks.db")
        self.s = TaskScheduler(self.db)

    def test_run_now_logs(self):
        self.s.add("t5", "每1小时", "hello")
        with patch("src.agent.chat", _fake_chat):
            out = self.s.run_now("t5")
        self.assertIn("✅", out)
        runs = self.s.recent_runs()
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0]["ok"])
        self.assertIn("自动任务产出", runs[0]["output"])

    def test_run_now_unknown(self):
        self.assertTrue(self.s.run_now("nope").startswith("⚠"))

    def test_run_now_disabled(self):
        self.s.add("t6", "每1小时", "x")
        self.s.set_enabled("t6", False)
        self.assertTrue(self.s.run_now("t6").startswith("⚠"))

    def test_kb_sink(self):
        fake_kb = _FakeKB()
        self.s.add("t7", "每1小时", "沉淀这条", kb_sink=True)
        with patch("src.agent.chat", _fake_chat), patch("src.tools._get_kb", lambda: fake_kb):
            self.s.run_now("t7")
        self.assertEqual(len(fake_kb.calls), 1)
        self.assertIn("t7", fake_kb.calls[0][0])


class TestBackgroundAndCatchUp(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "tasks.db")
        self.s = TaskScheduler(self.db)

    def _set_past(self, name):
        from datetime import datetime, timedelta
        past = (datetime.now() - timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S")
        with self.s._conn() as c:
            c.execute("UPDATE tasks SET next_run=? WHERE name=?", (past, name))

    def test_catch_up_on_start(self):
        self.s.add("bg1", "每1小时", "后台跑我")
        self._set_past("bg1")  # 模拟离线期间已到期
        with patch("src.agent.chat", _fake_chat):
            self.s.start()       # 启动 → _catch_up 补跑到期任务
            time.sleep(2.0)      # 等线程执行完
            self.s.stop()
        runs = self.s.recent_runs()
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0]["ok"])

    def test_auto_trigger_loop(self):
        # 直接把 next_run 设为过去，启动后第一轮循环应触发
        self.s.add("bg2", "每1小时", "循环触发")
        self._set_past("bg2")
        with patch("src.agent.chat", _fake_chat):
            self.s.start()
            time.sleep(2.0)
            self.s.stop()
        self.assertEqual(len(self.s.recent_runs()), 1)

    def test_no_trigger_when_future(self):
        self.s.add("bg3", "每1小时", "未来才跑")
        # next_run 默认是 now+1h，启动不应触发
        with patch("src.agent.chat", _fake_chat):
            self.s.start()
            time.sleep(1.5)
            self.s.stop()
        self.assertEqual(len(self.s.recent_runs()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
