"""#17 客户端执行器测试（全离线：不需要真被控 PC、不需要 GUI 依赖、不调模型、不写 repo）。

四层：
  · 中心枢纽：注册/心跳/离线/队列/超时/分阶段放权/审计（纯内存）
  · 客户端：能力执行（shell 走沙箱 / 文件 jail / 截屏与输入用假 driver）、客户端策略、结果回传
  · HTTP 端到端：真起 uvicorn + 真 ExecutorClient 长轮询 → 派命令 → 拿结果（真 socket、真鉴权）
  · 四角色 Computer Use：用注入的假角色调用验证「四个角色真的分开、评估者只看证据」等防幻觉机制
"""
import base64
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, ".")

import src.executor_hub as hubmod                              # noqa: E402
from src.cua import Action, CuaError, ComputerUseLoop, Plan, PlanStep, Replan, Verdict  # noqa: E402
from src.executor import ExecutorClient, FakeDriver, NullDriver  # noqa: E402
from src.executor_hub import ExecutorError, ExecutorHub, reset_hub  # noqa: E402
from src.logging_setup import init_logger, reset_logger        # noqa: E402

PY = sys.executable


class HubBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge-exec-")
        init_logger({"logging": {"dir": self.tmp, "level": "DEBUG", "enabled": True}})
        self.addCleanup(reset_logger)
        reset_hub()
        self.addCleanup(reset_hub)
        self.hub = ExecutorHub({"executor": {"hub": {"poll_wait": 0.2, "command_timeout": 1.0}}})

    def add_device(self, device_id="pc-01", caps=None, **kw):
        return self.hub.register({"device_id": device_id, "host": "HOST-A", "os": "Windows-10",
                                  "capabilities": caps or ["shell", "read_file", "list_dir", "screenshot"],
                                  **kw})


# ══════════════════════════════════════════════════════════════════
class TestHubRegistry(HubBase):
    def test_register_and_list(self):
        out = self.add_device()
        self.assertTrue(out["ok"])
        self.assertEqual(out["device"]["device_id"], "pc-01")
        self.assertEqual(out["hub_stage"], "low_risk")
        self.assertIn("shell", out["device"]["capabilities"])
        self.assertEqual(len(self.hub.devices()), 1)
        self.assertTrue(self.hub.devices()[0]["online"])

    def test_register_requires_device_id(self):
        with self.assertRaises(ExecutorError) as cm:
            self.hub.register({})
        self.assertEqual(cm.exception.code, "E_NO_DEVICE")

    def test_capabilities_filtered_to_known(self):
        out = self.add_device(caps=["shell", "核弹", "screenshot"])
        self.assertEqual(sorted(out["device"]["capabilities"]), ["screenshot", "shell"])

    def test_reregister_updates_and_touches(self):
        self.add_device(caps=["shell"])
        self.hub._devices["pc-01"].last_seen = time.time() - 999
        out = self.add_device(caps=["shell", "screenshot"])
        self.assertIn("screenshot", out["device"]["capabilities"])
        self.assertLess(self.hub.devices()[0]["last_seen_ago"], 5)

    def test_poll_unknown_device(self):
        with self.assertRaises(ExecutorError) as cm:
            self.hub.poll("没这台")
        self.assertEqual(cm.exception.code, "E_UNKNOWN_DEVICE")

    def test_prune_stale(self):
        self.add_device()
        self.hub._devices["pc-01"].last_seen = time.time() - 999
        self.assertEqual(self.hub.prune(), ["pc-01"])
        self.assertEqual(self.hub.devices(), [])


class TestHubDispatch(HubBase):
    def _pump(self, device_id="pc-01", handler=None, stop_after=3, wait=0.05):
        """起一个「假装是客户端」的线程：轮询领命令 → 回结果。"""
        done = {"n": 0}

        def loop():
            while done["n"] < stop_after:
                try:
                    cmds = self.hub.poll(device_id, wait=wait)
                except ExecutorError:
                    return
                for c in cmds:
                    out = handler(c) if handler else (True, "ok", "", {})
                    self.hub.submit_result(device_id, c["seq"], out[0], out[1], out[2], out[3])
                    done["n"] += 1

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        self.addCleanup(lambda: th.join(timeout=0.2))
        return done

    def test_dispatch_round_trip(self):
        self.add_device()
        self._pump()
        cmd = self.hub.dispatch("pc-01", "shell", "", {"command": "echo hi"})
        self.assertTrue(cmd.ok)
        self.assertEqual(cmd.output, "ok")
        self.assertEqual(cmd.status, "done")

    def test_dispatch_timeout(self):
        self.add_device()
        cmd = self.hub.dispatch("pc-01", "shell", "", {"command": "x"}, timeout=0.3)
        self.assertFalse(cmd.ok)
        self.assertEqual(cmd.status, "timeout")
        self.assertIn("超时", cmd.error)

    def test_dispatch_unknown_and_offline(self):
        with self.assertRaises(ExecutorError) as cm:
            self.hub.dispatch("ghost", "shell", "", {})
        self.assertEqual(cm.exception.code, "E_UNKNOWN_DEVICE")
        self.add_device()
        self.hub._devices["pc-01"].last_seen = time.time() - 999
        with self.assertRaises(ExecutorError) as cm2:
            self.hub.dispatch("pc-01", "shell", "", {})
        self.assertEqual(cm2.exception.code, "E_OFFLINE")

    def test_undisclosed_capability_refused(self):
        self.add_device(caps=["read_file"])
        with self.assertRaises(ExecutorError) as cm:
            self.hub.dispatch("pc-01", "shell", "", {"command": "x"})
        self.assertEqual(cm.exception.code, "E_CAP_UNSUPPORTED")

    def test_queue_full(self):
        self.hub.conf["max_queue"] = 1
        self.add_device()
        self.hub.dispatch("pc-01", "shell", "", {"command": "x"}, timeout=0.05)   # 占住队列
        with self.assertRaises(ExecutorError) as cm:
            self.hub.dispatch("pc-01", "shell", "", {"command": "y"}, timeout=0.05)
        self.assertEqual(cm.exception.code, "E_QUEUE_FULL")

    def test_stage_readonly_blocks_writes(self):
        hub = ExecutorHub({"executor": {"hub": {"stage": "readonly"}}})
        hub.register({"device_id": "pc-02", "capabilities": ["shell", "read_file"]})
        with self.assertRaises(ExecutorError) as cm:
            hub.dispatch("pc-02", "shell", "", {"command": "x"})
        self.assertEqual(cm.exception.code, "E_STAGE_READONLY")
        self.assertTrue(any(e.get("blocked") for e in hub.audit()))
        # 只读能力放行（直接派；没有客户端所以超时，但说明策略没拦）
        cmd = hub.dispatch("pc-02", "read_file", "", {"path": "a"}, timeout=0.1)
        self.assertEqual(cmd.status, "timeout")

    def test_stage_approval_needs_flag(self):
        hub = ExecutorHub({"executor": {"hub": {"stage": "approval"}}})
        hub.register({"device_id": "pc-03", "capabilities": ["shell", "read_file"]})
        with self.assertRaises(ExecutorError) as cm:
            hub.dispatch("pc-03", "shell", "", {"command": "x"})
        self.assertEqual(cm.exception.code, "E_NEED_APPROVAL")
        cmd = hub.dispatch("pc-03", "shell", "", {"command": "x"}, timeout=0.1, approved=True)
        self.assertEqual(cmd.status, "timeout")           # 放行了（无客户端才超时）

    def test_allow_caps_whitelist(self):
        hub = ExecutorHub({"executor": {"hub": {"allow_caps": ["read_file"]}}})
        hub.register({"device_id": "pc-04", "capabilities": ["shell", "read_file"]})
        with self.assertRaises(ExecutorError) as cm:
            hub.dispatch("pc-04", "shell", "", {"command": "x"})
        self.assertEqual(cm.exception.code, "E_CAP_DENIED")

    def test_audit_and_stats(self):
        self.add_device()
        self._pump(handler=lambda c: (False, "", "boom", {}))
        self.hub.dispatch("pc-01", "shell", "", {"command": "x"})
        rows = self.hub.audit(5)
        self.assertTrue(rows and rows[-1]["ok"] is False)
        st = self.hub.stats()
        self.assertEqual(st["devices"], 1)
        self.assertIn("stage", st)


# ══════════════════════════════════════════════════════════════════
class _FakeResult:
    def __init__(self, ok=True, output="ok", exit_code=0, mode="local", ms=3.0):
        self.ok, self.output, self.exit_code, self.mode, self.ms = ok, output, exit_code, mode, ms


class _FakeSandbox:
    def __init__(self, res=None):
        self.res = res or _FakeResult()
        self.commands = []

    def run(self, command, **kw):
        self.commands.append(command)
        return self.res


class _FakeHTTP:
    """假中心：记录请求，给预设响应。"""

    def __init__(self, responses=None):
        self.posts = []
        self.responses = responses or {}

    def post(self, path, json=None):
        self.posts.append((path, json))

        class R:
            status_code = 200

            def __init__(self, payload):
                self._p = payload

            def json(self):
                return self._p

            @property
            def text(self):
                return ""

        return R(self.responses.get(path, {"ok": True, "commands": []}))


class TestExecutorClient(HubBase):
    def _client(self, **kw):
        conf = {"root": self.tmp, "stage": "low_risk", "max_bytes": 1024}
        conf.update(kw.pop("client_conf", {}))
        return ExecutorClient(center="http://center.local", token="k", device_id="pc-01",
                              client_conf=conf, http=_FakeHTTP(), **kw)

    def test_capabilities_without_gui_deps(self):
        c = ExecutorClient(client_conf={"root": self.tmp}, driver=NullDriver())
        caps = c.capabilities()
        self.assertIn("shell", caps)
        self.assertNotIn("screenshot", caps)      # 没装 GUI 依赖 → 不声明（不假装有）

    def test_capabilities_with_fake_driver(self):
        c = self._client(driver=FakeDriver())
        caps = c.capabilities()
        self.assertIn("screenshot", caps)
        self.assertIn("input", caps)

    def test_shell_goes_through_sandbox(self):
        sb = _FakeSandbox(_FakeResult(True, "hello", 0, "local"))
        c = self._client(sandbox=sb)
        ok, out, err, extra = c.handle("shell", "", {"command": "echo hello"})
        self.assertTrue(ok)
        self.assertEqual(out, "hello")
        self.assertEqual(sb.commands, ["echo hello"])
        self.assertEqual(extra["mode"], "local")

    def test_file_read_write_list_and_jail(self):
        c = self._client(driver=FakeDriver())
        ok, out, err, extra = c.handle("write_file", "", {"path": "a/b.txt", "content": "你好"})
        self.assertTrue(ok, err)
        ok2, b64, err2, extra2 = c.handle("read_file", "", {"path": "a/b.txt"})
        self.assertTrue(ok2, err2)
        self.assertEqual(base64.b64decode(b64).decode("utf-8"), "你好")
        ok3, listing, _, _ = c.handle("list_dir", "", {"path": "a"})
        self.assertIn("b.txt", listing)
        # 越界路径
        ok4, _, err4, extra4 = c.handle("read_file", "", {"path": "../outside.txt"})
        self.assertFalse(ok4)
        self.assertIn("E_PATH_DENIED", err4)

    def test_size_limit(self):
        c = self._client(driver=FakeDriver(), client_conf={"max_bytes": 8})
        ok, _, err, _ = c.handle("write_file", "", {"path": "big.txt", "content": "x" * 100})
        self.assertFalse(ok)
        self.assertIn("E_TOO_LARGE", err)

    def test_screenshot_and_input_with_fake_driver(self):
        drv = FakeDriver(png=b"\x89PNG-fake", size=(800, 600))
        c = self._client(driver=drv)
        ok, out, err, extra = c.handle("screenshot", "", {})
        self.assertTrue(ok, err)
        self.assertEqual(base64.b64decode(out), b"\x89PNG-fake")
        self.assertEqual(extra["screen"], [800, 600])
        for action, args in (("click", {"x": 10, "y": 20}), ("move", {"x": 1, "y": 2}),
                             ("type", {"text": "hi"}), ("key", {"keys": "ctrl+s"}), ("scroll", {"amount": -3})):
            ok2, _, err2, _ = c.handle("input", action, args)
            self.assertTrue(ok2, f"{action} 失败：{err2}")
        self.assertEqual([a[0] for a in drv.actions], ["click", "move", "type", "key", "scroll"])
        ok3, _, err3, _ = c.handle("input", "dance", {})
        self.assertFalse(ok3)
        self.assertIn("E_BAD_ARGS", err3)

    def test_gui_capability_reports_unavailable_not_fake_success(self):
        c = self._client(driver=NullDriver("没装 pyautogui"))
        ok, _, err, _ = c.handle("screenshot", "", {})
        self.assertFalse(ok)                     # **绝不假装成功**
        self.assertIn("E_NO_GUI", err)
        ok2, _, err2, _ = c.handle("input", "click", {"x": 1, "y": 1})
        self.assertFalse(ok2)
        self.assertIn("E_NO_GUI", err2)

    def test_client_policy_readonly_and_approval(self):
        c = self._client(driver=FakeDriver(), client_conf={"stage": "readonly"})
        ok, _, err, extra = c.handle("shell", "", {"command": "echo x"})
        self.assertFalse(ok)
        self.assertIn("E_STAGE_READONLY", err)
        self.assertTrue(extra.get("blocked"))
        c2 = self._client(driver=FakeDriver(), client_conf={"stage": "approval"})
        ok2, _, err2, _ = c2.handle("write_file", "", {"path": "x.txt", "content": "1"})
        self.assertFalse(ok2)
        self.assertIn("E_NEED_APPROVAL", err2)
        ok3, _, err3, _ = c2.handle("write_file", "", {"path": "x.txt", "content": "1"}, approved=True)
        self.assertTrue(ok3, err3)

    def test_cap_whitelist(self):
        c = self._client(driver=FakeDriver(), client_conf={"allow_caps": ["read_file"]})
        ok, _, err, _ = c.handle("shell", "", {"command": "echo x"})
        self.assertFalse(ok)
        self.assertIn("E_CAP_DENIED", err)

    def test_run_command_posts_result_and_audits(self):
        http = _FakeHTTP()
        c = self._client(driver=FakeDriver(), sandbox=_FakeSandbox())
        c._http = http
        out = c.run_command({"seq": "s1", "cap": "shell", "action": "", "args": {"command": "echo hi"}})
        self.assertTrue(out["ok"])
        self.assertEqual(http.posts[-1][0], "/api/executor/result")
        self.assertEqual(http.posts[-1][1]["seq"], "s1")
        self.assertEqual(c.local_audit[-1]["cap"], "shell")

    def test_register_and_loop_re_registers_on_unknown(self):
        http = _FakeHTTP(responses={"/api/executor/poll": {"ok": True, "commands": []}})
        c = self._client(driver=FakeDriver())
        c._http = http
        c.register()
        c.loop(duration=0.05)
        self.assertEqual(http.posts[0][0], "/api/executor/register")
        self.assertTrue(any(p[0] == "/api/executor/poll" for p in http.posts))


# ══════════════════════════════════════════════════════════════════
try:
    import uvicorn                                            # noqa: E402

    from src.server import SessionStore, create_app            # noqa: E402
    HAS_SERVER = True
except Exception:                                              # pragma: no cover
    HAS_SERVER = False


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@unittest.skipUnless(HAS_SERVER, "未装服务依赖（pip install handcraft-agent[server]）")
class TestHttpEndToEnd(HubBase):
    """真 HTTP：uvicorn + 真 ExecutorClient 长轮询 → 派命令 → 拿结果。"""

    def _serve(self, keys=("k",)):
        port = _free_port()
        cfg = {"server": {"db_path": os.path.join(self.tmp, "s.db")},
               "executor": {"hub": {"poll_wait": 1.0, "command_timeout": 8.0}}}
        app = create_app(cfg=cfg, api_keys=list(keys), rate_limit=0,
                         store=SessionStore(os.path.join(self.tmp, "s.db")))
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error",
                                lifespan="off", access_log=False)
        server = uvicorn.Server(config)
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        import httpx

        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5).status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("服务没起来")

        def stop():
            server.should_exit = True
            th.join(timeout=3)

        self.addCleanup(stop)
        return port, app

    def test_full_round_trip_over_http(self):
        port, app = self._serve()
        client = ExecutorClient(center=f"http://127.0.0.1:{port}", token="k", device_id="pc-e2e",
                                client_conf={"root": self.tmp, "stage": "low_risk"}, driver=FakeDriver(),
                                verbose=False)
        client.register()
        th = threading.Thread(target=client.loop, kwargs={"duration": 12}, daemon=True)
        th.start()
        self.addCleanup(lambda: setattr(client, "stop_flag", True))
        hub = app.state.executor_hub
        deadline = time.time() + 5
        while time.time() < deadline and not hub.devices():
            time.sleep(0.05)
        self.assertTrue(hub.devices(), "执行器没注册上")
        cmd = hub.dispatch("pc-e2e", "shell", "", {"command": f'"{PY}" -c "print(6*7)"'}, timeout=8)
        self.assertTrue(cmd.ok, f"{cmd.status} {cmd.error} {cmd.output}")
        self.assertIn("42", cmd.output)
        self.assertEqual(cmd.extra.get("exit_code"), 0)

    def test_screenshot_and_policy_over_http(self):
        port, app = self._serve()
        client = ExecutorClient(center=f"http://127.0.0.1:{port}", token="k", device_id="pc-png",
                                client_conf={"root": self.tmp, "allow_caps": ["screenshot", "read_file"]},
                                driver=FakeDriver(png=b"PNGDATA"), verbose=False)
        client.register()
        th = threading.Thread(target=client.loop, kwargs={"duration": 12}, daemon=True)
        th.start()
        self.addCleanup(lambda: setattr(client, "stop_flag", True))
        hub = app.state.executor_hub
        deadline = time.time() + 5
        while time.time() < deadline and not hub.devices():
            time.sleep(0.05)
        cmd = hub.dispatch("pc-png", "screenshot", "", {}, timeout=8)
        self.assertTrue(cmd.ok, cmd.error)
        self.assertEqual(base64.b64decode(cmd.output), b"PNGDATA")
        # 未声明的能力 → 拒绝（执行器没声明 write_file 就不会被派）
        with self.assertRaises(ExecutorError) as cm:
            hub.dispatch("pc-png", "write_file", "", {"path": "x", "content": "1"})
        self.assertIn(cm.exception.code, ("E_CAP_UNSUPPORTED", "E_CAP_DENIED"))

    def test_http_auth_required(self):
        port, _app = self._serve(keys=("secret-key",))
        import httpx

        r = httpx.post(f"http://127.0.0.1:{port}/api/executor/register",
                       json={"device_id": "x", "capabilities": ["shell"]}, timeout=5)
        self.assertEqual(r.status_code, 401)
        r2 = httpx.get(f"http://127.0.0.1:{port}/api/executor/devices", timeout=5)
        self.assertEqual(r2.status_code, 401)
        r3 = httpx.get(f"http://127.0.0.1:{port}/api/executor/devices",
                       headers={"X-API-Key": "secret-key"}, timeout=5)
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.json()["devices"], [])


# ══════════════════════════════════════════════════════════════════
class _FakeCommand:
    def __init__(self, ok=True, output="done", error="", extra=None, status="done"):
        self.ok, self.output, self.error, self.status = ok, output, error, status
        self.extra = dict(extra or {})
        self.created_at = time.time()
        self.done_at = time.time()


class TestComputerUseFourRoles(HubBase):
    """四角色：用注入的假角色调用验证机制（不调模型）。"""

    def setUp(self):
        super().setUp()
        self.hub.register({"device_id": "pc-cua", "os": "Windows-10",
                           "capabilities": ["shell", "read_file", "screenshot"], "stage": "low_risk"})
        self.prompts = {}          # role → [user 文本]

        async def fake_caller(role_key, schema, system, user):
            self.prompts.setdefault(role_key, []).append(user)
            self.systems = getattr(self, "systems", {})
            self.systems[role_key] = system
            if role_key == "planner":
                return Plan(steps=[PlanStep(goal="跑一条命令"), PlanStep(goal="确认结果")], done_when="看到 42")
            if role_key == "executor":
                if len(self.prompts["executor"]) == 1:
                    return Action(cap="shell", action="", args={"command": "echo 42"}, expect="输出里有 42",
                                  why="执行者自己的小算盘-不该给评估者看")
                return Action(cap="done", action="", args={}, expect="已完成")
            if role_key == "evaluator":
                return Verdict(ok=True, reason="输出里确实有 42")
            return Replan(steps=[], give_up=False, reason="")

        self.caller = fake_caller

    def _loop(self, **conf):
        c = dict(conf)
        c.setdefault("max_steps", 5)
        return ComputerUseLoop({"cua": c}, hub=self.hub, caller=self.caller)

    def test_happy_path_uses_four_distinct_roles(self):
        loop = self._loop()
        with patch.object(self.hub, "dispatch", return_value=_FakeCommand(True, "42")):
            res = loop.run_sync("把 42 打出来", device_id="pc-cua")
        self.assertTrue(res.ok, res.reason)
        self.assertIn("planner", self.prompts)
        self.assertIn("executor", self.prompts)
        self.assertIn("evaluator", self.prompts)
        # 三/四个角色各自的 system prompt 不同（不是同一个「全能模型」）
        self.assertEqual(len(set(self.systems.values())), len(self.systems))
        roles_used = {c["role"] for c in loop.role_calls}
        self.assertLessEqual({"planner", "executor", "evaluator"}, roles_used)
        self.assertTrue(any(s.cap == "done" and s.ok for s in res.steps))

    def test_evaluator_never_sees_executor_private_reasoning(self):
        """防幻觉的关键：评估者只看原始证据，看不到执行者的 why（信息不对称）。"""
        loop = self._loop()
        with patch.object(self.hub, "dispatch", return_value=_FakeCommand(True, "42")):
            loop.run_sync("把 42 打出来", device_id="pc-cua")
        ev_text = "\n".join(self.prompts["evaluator"])
        self.assertIn("42", ev_text)                     # 看得到原始证据
        self.assertNotIn("执行者自己的小算盘", ev_text)     # 看不到执行者的私话

    def test_failure_leads_to_supervisor_replan_then_give_up(self):
        async def failing_caller(role_key, schema, system, user):
            self.prompts.setdefault(role_key, []).append(user)
            if role_key == "planner":
                return Plan(steps=[PlanStep(goal="做不到的事")])
            if role_key == "executor":
                return Action(cap="shell", action="", args={"command": "boom"}, expect="ok")
            if role_key == "evaluator":
                return Verdict(ok=False, reason="证据显示失败")
            return Replan(steps=[], give_up=True, reason="任务本身做不到")

        loop = ComputerUseLoop({"cua": {"max_steps": 5, "max_failures": 2, "max_replans": 1}},
                               hub=self.hub, caller=failing_caller)
        with patch.object(self.hub, "dispatch", return_value=_FakeCommand(False, "", "boom")):
            res = loop.run_sync("做不到的任务", device_id="pc-cua")
        self.assertFalse(res.ok)
        self.assertEqual(res.replans, 1)
        self.assertIn("监督者", res.reason)
        self.assertIn("supervisor", self.prompts)

    def test_step_cap_respected(self):
        async def many_steps(role_key, schema, system, user):
            if role_key == "planner":
                return Plan(steps=[PlanStep(goal=f"step{i}") for i in range(30)])
            if role_key == "executor":
                return Action(cap="shell", action="", args={"command": "echo x"})
            if role_key == "evaluator":
                return Verdict(ok=True, reason="ok")
            return Replan(steps=[])

        loop = ComputerUseLoop({"cua": {"max_steps": 3, "max_failures": 9}},
                               hub=self.hub, caller=many_steps)
        with patch.object(self.hub, "dispatch", return_value=_FakeCommand(True, "x")):
            res = loop.run_sync("无限步骤", device_id="pc-cua")
        self.assertFalse(res.ok)                          # 执行者从不说 done → 步数用尽
        self.assertLessEqual(len(res.steps), 3)
        self.assertIn("步数", res.reason)

    def test_unknown_device_gives_actionable_error(self):
        loop = self._loop()
        with self.assertRaises(CuaError) as cm:
            loop.run_sync("随便", device_id="没有这台")
        self.assertIn("executor_agent.py", str(cm.exception))

    def test_disabled_by_config(self):
        loop = ComputerUseLoop({"cua": {"enabled": False}}, hub=self.hub, caller=self.caller)
        with self.assertRaises(CuaError):
            loop.run_sync("x", device_id="pc-cua")

    def test_empty_task_rejected(self):
        with self.assertRaises(CuaError):
            self._loop().run_sync("   ", device_id="pc-cua")


if __name__ == "__main__":
    unittest.main(verbosity=2)
