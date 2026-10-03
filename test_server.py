"""#14 服务化测试：鉴权 / 限流 / 会话持久化 / API 端点 / 后台启动。

全部离线：Agent 用 FakeAgent（不调模型、不联网），断言只针对服务层行为。
未装服务依赖时整包 skip（pip install "handcraft-agent[server]"）。
运行：python test_server.py  或  pytest test_server.py
"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, ".")

from forge.server import (  # noqa: E402
    ForgeService,
    _strip_injected,
    HAS_FASTAPI,
    HAS_UVICORN,
    RateLimiter,
    SessionStore,
    check_auth,
    create_app,
    extract_key,
    identity_of,
    is_loopback,
    resolve_api_keys,
)

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None


# ══════════════════════════════════════════════════════════════════
class FakeAgent:
    """假 Agent：回显 + 记录调用次数；含一条 system 消息模拟真实结构。"""

    max_context_tokens = 8000

    def __init__(self, name="fake"):
        self.name = name
        self.messages = [{"role": "system", "content": "test-system"}]
        self.total_tokens = {"prompt": 10, "completion": 5}
        self.calls = 0

    async def run(self, task):
        self.calls += 1
        self.messages.append({"role": "user", "content": task})
        reply = "用户拒绝了该写操作" if "WRITE" in task else f"echo#{self.calls}: {task}"
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _context_tokens(self):
        return sum(len(m.get("content") or "") for m in self.messages)


def _svc(tmp, **kw):
    """在临时目录建一个带假 Agent 的服务（绝不写进仓库/知识库）。"""
    cfg = {"server": {"db_path": os.path.join(tmp, "sessions.db")},
           "models": {"m1": {"label": "M1", "base_url": "http://x/v1", "model": "m-1", "api_key": "k"}},
           "roles": {"default": {"model": "m1"}}}
    store = SessionStore(cfg["server"]["db_path"])
    return ForgeService(cfg=cfg, store=store, agent_factory=lambda sid: FakeAgent(sid), **kw)


# ══════════════════════════════════════════════════════════════════
class TestAuth(unittest.TestCase):
    def test_loopback_detect(self):
        for h in ("127.0.0.1", "127.0.0.5", "::1", "localhost", ""):
            self.assertTrue(is_loopback(h), h)
        for h in ("10.0.0.2", "192.168.1.9", "8.8.8.8"):
            self.assertFalse(is_loopback(h), h)

    def test_no_keys_loopback_only(self):
        ok, why = check_auth([], "", "127.0.0.1")
        self.assertTrue(ok, why)
        ok, why = check_auth([], "", "1.2.3.4")
        self.assertFalse(ok)
        self.assertIn("仅允许本机", why)

    def test_keys_required(self):
        self.assertFalse(check_auth(["k1"], "", "127.0.0.1")[0])   # 配了 key，本机也要验
        self.assertTrue(check_auth(["k1"], "k1", "1.2.3.4")[0])
        self.assertFalse(check_auth(["k1"], "k2", "1.2.3.4")[0])
        self.assertTrue(check_auth(["k1", "k2"], "k2", "1.2.3.4")[0])

    def test_extract_key(self):
        self.assertEqual(extract_key("Bearer abc", None), "abc")
        self.assertEqual(extract_key("bearer abc", None), "abc")
        self.assertEqual(extract_key("abc", None), "abc")
        self.assertEqual(extract_key(None, "xyz"), "xyz")
        self.assertEqual(extract_key(None, None), "")
        self.assertEqual(extract_key("Bearer abc", "xyz"), "xyz")  # X-API-Key 优先

    def test_resolve_keys_from_config_and_env(self):
        self.assertEqual(resolve_api_keys({"server": {"api_keys": ["a", "b"]}}), ["a", "b"])
        self.assertEqual(resolve_api_keys({"server": {"api_key": "a,b"}}), ["a", "b"])
        self.assertEqual(resolve_api_keys({"server": {"api_key": " a , ,b "}}), ["a", "b"])
        self.assertEqual(resolve_api_keys({}), [])
        os.environ["_FORGE_TEST_SECRET"] = "s3cret"
        self.assertEqual(resolve_api_keys({"server": {"api_keys": ["env:_FORGE_TEST_SECRET"]}}), ["s3cret"])
        os.environ["FORGE_API_KEY"] = "e1,e2"
        try:
            self.assertEqual(resolve_api_keys({"server": {"api_keys": ["a"]}}), ["a", "e1", "e2"])
        finally:
            os.environ.pop("FORGE_API_KEY")
            os.environ.pop("_FORGE_TEST_SECRET")

    def test_identity(self):
        self.assertTrue(identity_of(["k"], "k", "1.2.3.4").startswith("key:"))
        self.assertTrue(identity_of([], "", "1.2.3.4").startswith("ip:"))
        # 同一 key 不同客户端 → 同标识；不同 key → 不同标识
        self.assertEqual(identity_of(["k"], "k", "a"), identity_of(["k"], "k", "b"))
        self.assertNotEqual(identity_of(["k", "j"], "k"), identity_of(["k", "j"], "j"))


class TestRateLimiter(unittest.TestCase):
    def test_window(self):
        rl = RateLimiter(limit=3, window=60)
        self.assertEqual([rl.hit("u") for _ in range(3)], [0.0, 0.0, 0.0])
        self.assertGreater(rl.hit("u"), 0)          # 第 4 次被限
        self.assertEqual(rl.hit("other"), 0.0)      # 互不影响
        rl.reset("u")
        self.assertEqual(rl.hit("u"), 0.0)

    def test_expire_and_disable(self):
        rl = RateLimiter(limit=1, window=10)
        self.assertEqual(rl.hit("u", now=100), 0.0)
        self.assertGreater(rl.hit("u", now=101), 0)
        self.assertEqual(rl.hit("u", now=115), 0.0)   # 窗口滑过去了
        self.assertEqual(RateLimiter(limit=0).hit("u"), 0.0)  # 关闭限流


class TestSessionStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge-srv-")
        self.store = SessionStore(os.path.join(self.tmp, "s.db"))

    def test_crud(self):
        sid = self.store.create(title="第一会话")
        s = self.store.get(sid)
        self.assertEqual(s["title"], "第一会话")
        self.assertEqual(s["messages"], 0)
        self.assertEqual([x["id"] for x in self.store.list_sessions()], [sid])
        self.assertTrue(self.store.delete(sid))
        self.assertEqual(self.store.get(sid), {})
        self.assertFalse(self.store.delete(sid))

    def test_messages_roundtrip_and_persistence(self):
        sid = self.store.create()
        msgs = [{"role": "user", "content": "你好"},
                {"role": "assistant", "content": "hi", "tool_calls": [{"id": "t1"}]},
                {"role": "tool", "tool_call_id": "t1", "content": "42"}]
        self.store.save_messages(sid, msgs)
        self.assertEqual(self.store.history(sid), msgs)      # tool_calls 等结构原样保留
        self.assertEqual(self.store.get(sid)["messages"], 3)
        self.assertEqual(self.store.get(sid)["turns"], 1)
        # 换实例（同库）仍读得到 → 持久化
        store2 = SessionStore(os.path.join(self.tmp, "s.db"))
        self.assertEqual(len(store2.history(sid)), 3)

    def test_save_is_idempotent_overwrite(self):
        sid = self.store.create()
        self.store.save_messages(sid, [{"role": "user", "content": "a"}] * 5)
        self.store.save_messages(sid, [{"role": "user", "content": "b"}])
        self.assertEqual(len(self.store.history(sid)), 1)
        self.assertEqual(self.store.count(), 1)

    def test_unknown_session(self):
        self.assertEqual(self.store.get("nope"), {})
        self.assertEqual(self.store.history("nope"), [])


# ══════════════════════════════════════════════════════════════════
class TestInjectedStripping(unittest.TestCase):
    """记忆注入只作用于当次请求：写进历史前必须剥掉（否则历史被注入文本糊住）。"""

    def test_strip_user_only(self):
        injected = "[关于你的长期记忆]\n- 我是做跨境电商的"
        u = {"role": "user", "content": injected + "\n\n用户问题：我是谁"}
        self.assertEqual(_strip_injected(u, injected)["content"], "我是谁")
        # 非注入消息 / 非 user 角色 / 无前缀 → 原样
        self.assertEqual(_strip_injected({"role": "user", "content": "普通提问"}, injected)["content"],
                         "普通提问")
        a = {"role": "assistant", "content": injected}
        self.assertEqual(_strip_injected(a, injected), a)
        self.assertEqual(_strip_injected(u, "")["content"], u["content"])
        # 不修改原对象（避免污染 Agent 内存里的历史）
        self.assertIn("用户问题", u["content"])


@unittest.skipUnless(HAS_FASTAPI and TestClient, "未装服务依赖：pip install handcraft-agent[server]")
class TestApi(unittest.TestCase):
    KEY = "test-key"
    H = {"Authorization": "Bearer test-key"}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge-api-")
        # 隔离长期记忆库（否则测试会写进 forge 真实的 data/memory.db）
        import forge.memory as memory_mod
        from forge.memory import MemoryStore

        self._orig_get_memory = memory_mod.get_memory
        memory_mod.get_memory = lambda: MemoryStore(os.path.join(self.tmp, "memory.db"))
        self.addCleanup(lambda: setattr(memory_mod, "get_memory", self._orig_get_memory))
        self.svc = _svc(self.tmp)
        self.app = create_app(cfg={"server": {}}, service=self.svc, api_keys=[self.KEY],
                              rate_limit=0)
        self.c = TestClient(self.app)

    def test_healthz_no_auth(self):
        r = self.c.get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_status_requires_key(self):
        self.assertEqual(self.c.get("/api/status").status_code, 401)
        self.assertEqual(self.c.get("/api/status", headers={"X-API-Key": "wrong"}).status_code, 401)
        r = self.c.get("/api/status", headers={"X-API-Key": self.KEY})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["auth"], "api_key")
        self.assertEqual(body["model"], "m-1")
        self.assertEqual(body["sessions"], 0)

    def test_chat_autocreates_session_and_reports_usage(self):
        r = self.c.post("/api/chat", json={"message": "你好"}, headers=self.H)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["session_id"])
        self.assertIn("echo#1", body["reply"])
        self.assertEqual(body["usage"]["prompt"], 10)
        self.assertIsNotNone(body["usage"]["context_pct"])
        self.assertEqual(self.svc.store.count(), 1)

    def test_chat_continues_session_and_persists_history(self):
        sid = self.c.post("/api/chat", json={"message": "第一轮"}, headers=self.H).json()["session_id"]
        self.c.post("/api/chat", json={"message": "第二轮", "session_id": sid}, headers=self.H)
        hist = self.svc.store.history(sid)
        self.assertEqual([m["role"] for m in hist], ["user", "assistant", "user", "assistant"])
        self.assertTrue(all(m["role"] != "system" for m in hist))   # system 不进库
        self.assertIn("第二轮", hist[-2]["content"])

    def test_history_restored_for_new_service_instance(self):
        sid = self.c.post("/api/chat", json={"message": "记住这个"}, headers=self.H).json()["session_id"]
        svc2 = _svc(self.tmp)                       # 模拟服务重启（同库新服务）
        app2 = create_app(cfg={"server": {}}, service=svc2, api_keys=[self.KEY], rate_limit=0)
        c2 = TestClient(app2)
        r = c2.post("/api/chat", json={"message": "接着聊", "session_id": sid}, headers=self.H)
        self.assertEqual(r.status_code, 200)
        agent = svc2._agents[sid]
        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertEqual(agent.messages[1]["role"], "user")
        # 历史恢复的是**用户原文**：记忆注入的前缀不落库（只作用于当次请求）
        self.assertEqual(agent.messages[1]["content"], "记住这个")
        self.assertIn("记住这个", agent.messages[2]["content"])

    def test_chat_validation(self):
        self.assertEqual(self.c.post("/api/chat", json={}, headers=self.H).status_code, 400)
        self.assertEqual(self.c.post("/api/chat", json={"message": "   "}, headers=self.H).status_code, 400)
        r = self.c.post("/api/chat", json={"message": "hi", "session_id": "nope"}, headers=self.H)
        self.assertEqual(r.status_code, 404)

    def test_sessions_crud(self):
        sid = self.c.post("/api/sessions", json={"title": "演示"}, headers=self.H).json()["session_id"]
        self.assertEqual(self.c.get("/api/sessions", headers=self.H).json()["sessions"][0]["title"], "演示")
        detail = self.c.get(f"/api/sessions/{sid}", headers=self.H).json()
        self.assertEqual(detail["messages"], [])
        self.assertEqual(self.c.delete(f"/api/sessions/{sid}", headers=self.H).status_code, 200)
        self.assertEqual(self.c.get(f"/api/sessions/{sid}", headers=self.H).status_code, 404)
        self.assertEqual(self.c.delete(f"/api/sessions/{sid}", headers=self.H).status_code, 404)

    def test_write_rejected_warning_once(self):
        r1 = self.c.post("/api/chat", json={"message": "WRITE 文件"}, headers=self.H).json()
        self.assertIn("warning", r1)
        self.assertIn("写操作默认被拒", r1["warning"])
        r2 = self.c.post("/api/chat", json={"message": "WRITE 文件"}, headers=self.H).json()
        self.assertNotIn("warning", r2)              # 提示只给一次，不刷屏

    def test_unknown_path_404(self):
        self.assertEqual(self.c.get("/api/nope", headers=self.H).status_code, 404)

    def test_rate_limit_http(self):
        app = create_app(cfg={"server": {}}, service=self.svc, api_keys=[self.KEY], rate_limit=2)
        c = TestClient(app)
        self.assertEqual(c.get("/api/status", headers=self.H).status_code, 200)
        self.assertEqual(c.get("/api/status", headers=self.H).status_code, 200)
        r = c.get("/api/status", headers=self.H)
        self.assertEqual(r.status_code, 429)
        self.assertIn("Retry-After", r.headers)

    def test_no_key_mode_loopback_only(self):
        app = create_app(cfg={"server": {}}, service=self.svc, api_keys=[], rate_limit=0)
        try:
            c = TestClient(app, client=("127.0.0.1", 12345))
        except TypeError:            # 老 starlette 不支持 client= → 跳过（纯函数测试已覆盖）
            self.skipTest("TestClient 不支持 client= 参数")
        self.assertEqual(c.get("/api/status").status_code, 200)
        c2 = TestClient(app, client=("8.8.8.8", 12345))
        self.assertEqual(c2.get("/api/status").status_code, 401)


@unittest.skipUnless(HAS_FASTAPI and HAS_UVICORN and TestClient,
                     "未装服务依赖：pip install handcraft-agent[server]")
class TestBackgroundServe(unittest.TestCase):
    def test_start_and_stop(self):
        import httpx

        tmp = tempfile.mkdtemp(prefix="forge-bg-")
        cfg = {"server": {"db_path": os.path.join(tmp, "s.db"), "api_key": "k"}}
        from forge.server import SessionStore, start_background

        server, url = start_background(host="127.0.0.1", port=0, cfg=cfg,
                                       store=SessionStore(os.path.join(tmp, "s.db")))
        try:
            deadline = time.time() + 10
            ok = False
            while time.time() < deadline:
                try:
                    r = httpx.get(f"{url}healthz", timeout=2)
                    if r.status_code == 200 and r.json()["ok"]:
                        ok = True
                        break
                except Exception:
                    time.sleep(0.2)
            self.assertTrue(ok, "后台服务未在 10s 内就绪")
        finally:
            server.should_exit = True
            for _ in range(50):
                if getattr(server, "started", None) is False or not server.should_exit:
                    break
                time.sleep(0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
