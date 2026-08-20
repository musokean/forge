"""#5 熔断单元测试：三态机 + 集成进 llm.chat / stream_chat 的跳过与降级。

运行：python test_circuit_breaker.py
依赖：managed venv（openai / src 可 import）
"""
import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, ".")

from src.circuit import (
    CircuitBreaker,
    CircuitRegistry,
    CircuitOpenError,
    get_circuit_registry,
    reset_circuit_registry,
)


# ===================== 0) 全局隔离 =====================
class _Isolated(unittest.TestCase):
    """每个测试后复位熔断器注册表——防止污染后续测试文件
    （2026-08-20：CI 里 test_circuit_breaker 熔断角色 b 后，
     test_error_resilience 的降级测试全被「熔断中」拦截）。"""

    def tearDown(self):
        reset_circuit_registry()
        super().tearDown()


# ===================== 1) 单熔断器三态机 =====================
class TestBreakerStates(_Isolated):
    def test_closed_to_open_on_threshold(self):
        b = CircuitBreaker("x", failure_threshold=3, cooldown=30, clock=lambda: 1000)
        self.assertEqual(b.state, "closed")
        b.record_failure()
        b.record_failure()
        self.assertEqual(b.state, "closed")  # 2 < 3，仍关闭
        b.record_failure()
        self.assertEqual(b.state, "open")
        self.assertEqual(b.opened_at, 1000)

    def test_open_blocks_until_cooldown_then_half_open(self):
        b = CircuitBreaker("x", failure_threshold=1, cooldown=30, clock=lambda: 1000)
        b.record_failure()  # -> open at 1000
        self.assertTrue(b.is_open)
        self.assertFalse(b.allow())  # 1000 < 1030 冷却中
        b._clock = lambda: 1030  # 推进到冷却边界
        self.assertTrue(b.allow())  # -> half_open，放行探测
        self.assertEqual(b.state, "half_open")

    def test_half_open_recovers_on_success(self):
        b = CircuitBreaker("x", failure_threshold=1, cooldown=30, clock=lambda: 1000)
        b.record_failure()
        b._clock = lambda: 1030
        self.assertTrue(b.allow())  # half_open
        b.record_success()
        self.assertEqual(b.state, "closed")
        self.assertEqual(b.failures, 0)

    def test_half_open_reopens_on_failure(self):
        b = CircuitBreaker("x", failure_threshold=1, cooldown=30, clock=lambda: 1000)
        b.record_failure()
        b._clock = lambda: 1030
        self.assertTrue(b.allow())
        b.record_failure()
        self.assertEqual(b.state, "open")

    def test_reset_clears(self):
        b = CircuitBreaker("x", failure_threshold=1, cooldown=30, clock=lambda: 1000)
        b.record_failure()
        self.assertEqual(b.state, "open")
        b.reset()
        self.assertEqual(b.state, "closed")
        self.assertFalse(b.is_open)

    def test_success_clears_failure_count(self):
        b = CircuitBreaker("x", failure_threshold=3, cooldown=30, clock=lambda: 1000)
        b.record_failure()
        b.record_failure()
        self.assertEqual(b.failures, 2)
        b.record_success()
        self.assertEqual(b.failures, 0)
        self.assertEqual(b.state, "closed")


# ===================== 2) 注册表单例 =====================
class TestRegistry(_Isolated):
    def test_singleton_and_config_defaults(self):
        reset_circuit_registry()
        r1 = get_circuit_registry({"circuit_breaker": {"failure_threshold": 2, "cooldown": 10}})
        r2 = get_circuit_registry()
        self.assertIs(r1, r2)  # 单例
        b = r1.get("default")
        self.assertEqual(b.failure_threshold, 2)
        self.assertEqual(b.cooldown, 10)

    def test_reset_scoped(self):
        reset_circuit_registry()
        reg = get_circuit_registry({"circuit_breaker": {"failure_threshold": 1, "cooldown": 30}})
        reg.get("default").record_failure()
        self.assertEqual(reg.get("default").state, "open")
        reg.reset("default")
        self.assertEqual(reg.get("default").state, "closed")
        # 不影响另一个角色
        reg.get("fallback").record_failure()
        self.assertEqual(reg.get("fallback").state, "open")
        reg.reset()
        self.assertEqual(reg.get("fallback").state, "closed")


# ===================== 3) 集成：chat / stream_chat =====================
_CGFG = {
    "roles": {"default": {"model": "bad"}, "fallback": {"model": "good"}},
    "models": {
        "bad": {"base_url": "http://bad", "api_key": "kb", "model": "m"},
        "good": {"base_url": "http://good", "api_key": "kg", "model": "m"},
    },
    "circuit_breaker": {"failure_threshold": 2, "cooldown": 30},
}


class TestChatIntegration(_Isolated, unittest.IsolatedAsyncioTestCase):
    async def test_breaker_skips_failed_role_then_fallback(self):
        reset_circuit_registry()
        from src import llm

        async def fake_call(c, messages, role, tools=None, retries=3):
            if role == "default":
                raise RuntimeError("boom")
            return f"ok-from-{role}"

        with patch.object(llm, "_call_role", side_effect=fake_call):
            # 第 1、2 次 default 失败但没到阈值(2)，每次都降级到 fallback 成功
            r1 = await llm.chat(_CGFG, [{"role": "user", "content": "hi"}], role="default")
            r2 = await llm.chat(_CGFG, [{"role": "user", "content": "hi"}], role="default")
            self.assertEqual(r1, "ok-from-good")
            self.assertEqual(r2, "ok-from-good")
            # 第 2 次后 default 累计失败 2 次 → 熔断 OPEN
            reg = get_circuit_registry(_CGFG)
            self.assertEqual(reg.get("default").state, "open")
            # 第 3 次：default 熔断中直接跳过，fallback 仍成功（不再傻等重试）
            r3 = await llm.chat(_CGFG, [{"role": "user", "content": "hi"}], role="default")
            self.assertEqual(r3, "ok-from-good")

    async def test_all_roles_open_raises_circuit_open(self):
        reset_circuit_registry()
        cfg = {
            "roles": {"default": {"model": "bad"}, "fallback": {"model": "bad2"}},
            "models": {
                "bad": {"base_url": "http://bad", "api_key": "kb", "model": "m"},
                "bad2": {"base_url": "http://bad2", "api_key": "k2", "model": "m"},
            },
            "circuit_breaker": {"failure_threshold": 1, "cooldown": 30},
        }
        from src import llm

        async def fake_call(c, messages, role, tools=None, retries=3):
            raise RuntimeError("boom")

        with patch.object(llm, "_call_role", side_effect=fake_call):
            # 第一次：两角色都失败 → 都 OPEN → 抛原始 RuntimeError（fallback 的）
            with self.assertRaises(RuntimeError):
                await llm.chat(cfg, [{"role": "user", "content": "hi"}], role="default")
            reg = get_circuit_registry(cfg)
            self.assertEqual(reg.get("default").state, "open")
            # 注意：_fallback_role 对 dict 写法返回 model 字段值 "bad2"，而非 "fallback" 键名
            self.assertEqual(reg.get("bad2").state, "open")
            # 第二次：两角色都熔断中 → 全部跳过 → 抛 CircuitOpenError
            with self.assertRaises(CircuitOpenError):
                await llm.chat(cfg, [{"role": "user", "content": "hi"}], role="default")


class TestStreamChatIntegration(_Isolated, unittest.IsolatedAsyncioTestCase):
    async def test_stream_fallback_and_breaker(self):
        reset_circuit_registry()
        cfg = {
            "roles": {"default": {"model": "bad"}, "fallback": {"model": "good"}},
            "models": {
                "bad": {"base_url": "http://bad", "api_key": "kb", "model": "m"},
                "good": {"base_url": "http://good", "api_key": "kg", "model": "m"},
            },
            "circuit_breaker": {"failure_threshold": 1, "cooldown": 30},
        }
        from src import llm

        bad_client = AsyncMock()
        bad_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("conn"))

        async def good_create(**kw):
            async def gen():
                yield type(
                    "C", (), {
                        "choices": [type("Ch", (), {"delta": type("D", (), {"content": "ok-fb"})()})()],
                        "usage": None,
                    }
                )()
            return gen()

        good_client = AsyncMock()
        good_client.chat.completions.create = AsyncMock(side_effect=good_create)

        def fake_get_client(url, key):
            return bad_client if "bad" in url else good_client

        with patch.object(llm, "_get_client", side_effect=fake_get_client):
            collected = []
            async for d, u in llm.stream_chat(cfg, [{"role": "user", "content": "hi"}], role="default"):
                collected.append(d)
            self.assertTrue(
                any(getattr(x, "content", None) == "ok-fb" for x in collected),
                f"fallback 流式结果未收到，收到：{collected}",
            )
            reg = get_circuit_registry(cfg)
            self.assertEqual(reg.get("default").state, "open")  # default 已熔断


# ===================== 4) CLI 命令 =====================
class TestCli(_Isolated):
    def test_circuit_command_shows_open(self):
        reset_circuit_registry()
        reg = get_circuit_registry({"circuit_breaker": {"failure_threshold": 1, "cooldown": 30}})
        reg.get("default").record_failure()  # -> open
        from main import _circuit_command
        with patch("builtins.print") as p:
            _circuit_command("")
            out = " ".join(str(c.args[0]) for c in p.call_args_list)
        self.assertIn("OPEN", out)

    def test_circuit_command_reset(self):
        reset_circuit_registry()
        reg = get_circuit_registry({"circuit_breaker": {"failure_threshold": 1, "cooldown": 30}})
        reg.get("default").record_failure()
        self.assertEqual(reg.get("default").state, "open")
        from main import _circuit_command
        _circuit_command("reset")
        self.assertEqual(reg.get("default").state, "closed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
