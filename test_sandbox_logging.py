"""#4 沙箱 + #7 完整日志 的测试（全离线：Docker 用注入的假 runner，绝不真跑容器）。

覆盖：
  · 沙箱策略解析（auto/local/docker/off）、Docker 缺失时强制 docker → 拒绝执行
  · docker run 参数逐条断言（断网/限资源/只读挂载/非 root/环境白名单）
  · 超时（容器与本地）、输出截断、危险命令拦截
  · **本地加固的关键收益：宿主密钥不再透传给子进程**
  · 结构化日志：JSONL / 级别过滤 / 轮转 / 保留期 / **脱敏（key 不落盘）** / /logs 读取
  · 集成：run_command → 沙箱 → 日志一条；HTTP 请求 → 日志一条
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, ".")

import forge.logging_setup as logging_setup  # noqa: E402
import forge.sandbox as sb  # noqa: E402
from forge.logging_setup import Logger, get_logger, init_logger, log_event, redact, reset_logger  # noqa: E402
from forge.sandbox import (  # noqa: E402
    DockerSandbox,
    LocalSandbox,
    Sandbox,
    SandboxResult,
    SandboxUnavailable,
    docker_available,
    get_sandbox,
    reset_sandbox,
)

PY = sys.executable


class _FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _FakeRunner:
    """假 docker：记录 argv，返回预设结果；可模拟超时。"""

    def __init__(self, stdout="out", stderr="", returncode=0, timeout=False, kill_log=None):
        self.calls = []
        self.timeout = timeout
        self.kill_log = kill_log if kill_log is not None else []
        self._res = _FakeProc(stdout, stderr, returncode)

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[:2] == ["docker", "rm"]:
            self.kill_log.append(argv)
            return _FakeProc()
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kw.get("timeout", 1))
        return self._res


# ══════════════════════════════════════════════════════════════════
class TestSandboxPolicy(unittest.TestCase):
    def setUp(self):
        sb._reset_docker_cache()
        reset_sandbox()
        addCleanup = self.addCleanup
        addCleanup(reset_sandbox)
        addCleanup(sb._reset_docker_cache)

    def test_auto_falls_back_to_local_without_docker(self):
        with patch.object(sb, "docker_available", return_value=False):
            s = Sandbox(cfg={"sandbox": {"mode": "auto"}})
            self.assertEqual(s.resolve_mode(), "local")

    def test_auto_uses_docker_when_available(self):
        with patch.object(sb, "docker_available", return_value=True):
            s = Sandbox(cfg={"sandbox": {"mode": "auto"}})
            self.assertEqual(s.resolve_mode(), "docker")

    def test_forced_docker_without_docker_refuses(self):
        """mode=docker 且无 Docker → 拒绝执行（不静默降级到本机）。"""
        with patch.object(sb, "docker_available", return_value=False):
            s = Sandbox(cfg={"sandbox": {"mode": "docker"}})
            with self.assertRaises(SandboxUnavailable) as cm:
                s.run("echo 不该被执行")
            self.assertIn("拒绝在本机直接执行", str(cm.exception))

    def test_off_mode_runs_without_filtering(self):
        s = Sandbox(cfg={"sandbox": {"mode": "off"}}, workdir=tempfile.mkdtemp())
        res = s.run(f'"{PY}" -c "print(1+1)"')
        self.assertTrue(res.ok, res.output)
        self.assertEqual(res.mode, "off")
        self.assertIn("2", res.output)

    def test_status_fields(self):
        with patch.object(sb, "docker_available", return_value=False):
            st = Sandbox(cfg={"sandbox": {"mode": "auto"}}).status()
        for k in ("configured_mode", "effective_mode", "docker_available", "image", "limits", "deny_patterns"):
            self.assertIn(k, st)
        self.assertEqual(st["effective_mode"], "local")
        self.assertFalse(st["limits"]["network"])

    def test_singleton_refresh_reads_config(self):
        reset_sandbox()
        s1 = get_sandbox({"sandbox": {"mode": "local"}})
        self.assertEqual(s1.mode, "local")
        self.assertIs(get_sandbox(), s1)                       # 默认复用单例
        s2 = get_sandbox({"sandbox": {"mode": "off"}}, refresh=True)
        self.assertEqual(s2.mode, "off")                       # refresh 后按新配置


class TestDockerArgv(unittest.TestCase):
    """容器参数逐条断言——沙箱的隔离强度全在这里。"""

    def _argv(self, **kw):
        with patch.object(sb, "docker_available", return_value=True):
            d = DockerSandbox(workdir="/host/work", **kw)
            return d.build_argv("echo hi", "forge-sbx-test")

    def test_isolation_flags(self):
        argv = self._argv()
        self.assertEqual(argv[:2], ["docker", "run"])
        for flag in ("--rm", "--read-only"):
            self.assertIn(flag, argv)
        for flag, val in (("--network", "none"), ("--user", "65534:65534"), ("--pids-limit", "128"),
                          ("--memory", "256m"), ("--cpus", "1.0"), ("--tmpfs", "/tmp:rw,size=64m")):
            self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index(flag) + 1], val)
        # 工作目录只读挂载
        self.assertIn("-v", argv)
        self.assertIn("/host/work:/work:ro", argv)
        self.assertEqual(argv[argv.index("-w") + 1], "/work")
        # 命令通过 sh -c 交给容器
        self.assertEqual(argv[-4:-1], ["python:3.11-slim", "sh", "-c"])
        self.assertEqual(argv[-1], "echo hi")

    def test_network_and_rw_switches(self):
        argv = self._argv(network=True)
        self.assertEqual(argv[argv.index("--network") + 1], "bridge")
        argv2 = self._argv(mount_rw=True)
        self.assertIn("/host/work:/work", argv2)               # 无 :ro
        argv3 = self._argv(read_only_root=False)
        self.assertNotIn("--read-only", argv3)

    def test_env_passthrough_only_listed(self):
        os.environ["_SBX_OK"] = "yes"
        os.environ["_SBX_SECRET"] = "should-not-pass"
        try:
            argv = self._argv(env_passthrough=["_SBX_OK"])
            self.assertIn("-e", argv)
            self.assertIn("_SBX_OK=yes", argv)
            self.assertNotIn("_SBX_SECRET=should-not-pass", argv)
        finally:
            os.environ.pop("_SBX_OK", None)
            os.environ.pop("_SBX_SECRET", None)

    def test_run_result_and_nonzero(self):
        with patch.object(sb, "docker_available", return_value=True):
            runner = _FakeRunner(stdout="hello\n", stderr="warn", returncode=0)
            d = DockerSandbox(runner=runner)
            res = d.run("echo hi")
            self.assertTrue(res.ok)
            self.assertEqual(res.mode, "docker")
            self.assertIn("hello", res.output)
            self.assertIn("warn", res.output)
            self.assertEqual(runner.calls[0][-1], "echo hi")

            runner2 = _FakeRunner(stdout="", stderr="boom", returncode=3)
            res2 = DockerSandbox(runner=runner2).run("false")
            self.assertFalse(res2.ok)
            self.assertEqual(res2.exit_code, 3)

    def test_timeout_kills_container(self):
        with patch.object(sb, "docker_available", return_value=True):
            killed = []
            d = DockerSandbox(runner=_FakeRunner(timeout=True), deny_patterns=[])
            with patch("forge.sandbox.subprocess.run", side_effect=lambda argv, **kw: killed.append(argv) or _FakeProc()):
                res = d.run("sleep 999", timeout=1)
            self.assertFalse(res.ok)
            self.assertEqual(res.error, "timeout")
            self.assertIn("超时", res.output)
            self.assertTrue(any(a[:2] == ["docker", "rm"] for a in killed), killed)

    def test_dangerous_command_blocked_before_docker(self):
        with patch.object(sb, "docker_available", return_value=True):
            runner = _FakeRunner()
            res = DockerSandbox(runner=runner).run("rm -rf /")
            self.assertFalse(res.ok)
            self.assertEqual(res.error, "denied")
            self.assertEqual(runner.calls, [])                 # 根本没起容器


class TestLocalSandbox(unittest.TestCase):
    def test_runs_echo(self):
        res = LocalSandbox(cwd=tempfile.mkdtemp()).run(f'"{PY}" -c "print(\'ok-local\')"')
        self.assertTrue(res.ok, res.output)
        self.assertIn("ok-local", res.output)
        self.assertEqual(res.exit_code, 0)

    def test_blocked_patterns(self):
        ls = LocalSandbox(cwd=tempfile.mkdtemp())
        for cmd in ("rm -rf /", "mkfs.ext4 /dev/sda1", "shutdown -h now",
                    "dd if=/dev/zero of=/dev/sda"):
            res = ls.run(cmd)
            self.assertFalse(res.ok, cmd)
            self.assertIn("⛔", res.output)

    def test_secret_not_leaked_to_child(self):
        """关键收益：宿主 API key 不再交给子进程（旧实现 os.environ 整体透传）。"""
        os.environ["FORGE_TEST_SECRET"] = "sk-should-never-leak"
        try:
            code = "import os;print(os.environ.get('FORGE_TEST_SECRET','MISSING'))"
            res = LocalSandbox(cwd=tempfile.mkdtemp()).run(f'"{PY}" -c "{code}"')
        finally:
            os.environ.pop("FORGE_TEST_SECRET", None)
        self.assertIn("MISSING", res.output)
        self.assertNotIn("sk-should-never-leak", res.output)

    def test_timeout_and_truncate(self):
        res = LocalSandbox(timeout=1, cwd=tempfile.mkdtemp()).run(f'"{PY}" -c "import time;time.sleep(5)"')
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "timeout")
        res2 = LocalSandbox(max_output=10, cwd=tempfile.mkdtemp()).run(f'"{PY}" -c "print(\'x\'*500)"')
        self.assertIn("已截断", res2.output)

    def test_sandbox_result_dict(self):
        d = SandboxResult(True, "ok", 0, "local", 12.34).as_dict()
        self.assertEqual(d["mode"], "local")
        self.assertEqual(d["exit_code"], 0)
        self.assertAlmostEqual(d["ms"], 12.3, places=1)


class TestSandboxLoggingIntegration(unittest.TestCase):
    def test_run_command_goes_through_sandbox_and_logs(self):
        tmp = tempfile.mkdtemp(prefix="forge-sbxlog-")
        init_logger({"logging": {"dir": tmp, "level": "INFO", "enabled": True}})
        self.addCleanup(reset_logger)
        self.addCleanup(reset_sandbox)
        reset_sandbox()
        with patch.object(sb, "docker_available", return_value=False):
            from forge import tools

            out = tools.run_command(f'"{PY}" -c "print(\'via-sandbox\')"')
            self.assertIn("via-sandbox", out)
            self.assertIn("(命令无输出)", tools.run_command(f'"{PY}" -c "pass"'))
        recs = [r for r in get_logger().read(limit=20) if r["event"] == "sandbox_run"]
        self.assertTrue(recs, "sandbox_run 未落日志")
        self.assertEqual(recs[-1]["mode"], "local")
        self.assertIn("exit_code", recs[-1])

    def test_run_command_blocks_dangerous(self):
        reset_sandbox()
        with patch.object(sb, "docker_available", return_value=False):
            from forge import tools

            self.assertIn("拦截", tools.run_command("rm -rf /"))


# ══════════════════════════════════════════════════════════════════
class TestLogger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge-log-")
        self.cfg = {"logging": {"dir": self.tmp, "level": "INFO", "keep_days": 14, "max_mb": 20}}
        init_logger(self.cfg)
        self.addCleanup(reset_logger)

    def _lines(self, log=None):
        log = log or get_logger()
        out = []
        for p in log.files():
            with open(p, "r", encoding="utf-8") as f:
                out += [json.loads(l) for l in f if l.strip()]
        return out

    def test_jsonl_shape(self):
        log = get_logger()
        log.info("run_start", run_id="abc123", role="default", task="你好")
        rows = self._lines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "run_start")
        self.assertEqual(rows[0]["level"], "INFO")
        self.assertEqual(rows[0]["run_id"], "abc123")
        self.assertIn("ts", rows[0])
        self.assertTrue(log.files()[0].endswith(".jsonl"))

    def test_level_filter(self):
        log = get_logger()
        log.debug("noise")
        log.info("kept")
        log.error("boom")
        self.assertEqual([r["event"] for r in self._lines()], ["kept", "boom"])
        log.level = "ERROR"
        log.info("dropped")
        log.error("kept2")
        self.assertEqual([r["event"] for r in self._lines()][-1], "kept2")
        self.assertNotIn("dropped", [r["event"] for r in self._lines()])

    def test_disabled_writes_nothing(self):
        init_logger({"logging": {"dir": os.path.join(self.tmp, "off"), "enabled": False}})
        log = get_logger()
        log.info("should-not-write")
        self.assertEqual(log.files(), [])

    def test_rotation_by_size(self):
        init_logger({"logging": {"dir": os.path.join(self.tmp, "rot"), "max_mb": 0.0005, "level": "DEBUG"}})
        log = get_logger()
        for i in range(80):
            log.info("evt", i=i, pad="x" * 120)
        names = [os.path.basename(p) for p in log.files()]
        self.assertTrue(any("." in n.replace(".jsonl", "") for n in names), names)
        self.assertGreaterEqual(len(names), 2)

    def test_retention_prunes_old(self):
        d = os.path.join(self.tmp, "ret")
        os.makedirs(d)
        old = os.path.join(d, "forge-20200101.jsonl")
        new = os.path.join(d, "forge-20990101.jsonl")
        for p in (old, new):
            open(p, "w", encoding="utf-8").write('{"event":"x"}\n')
        past = time.time() - 40 * 86400
        os.utime(old, (past, past))
        log = Logger(log_dir=d, keep_days=14)
        self.assertEqual(log.prune(), 1)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(new))
        self.assertEqual(Logger(log_dir=d, keep_days=0).prune(), 0)   # 0 = 不清理

    def test_redaction(self):
        log = get_logger()
        log.info("llm_call", api_key="sk-abcdef1234567890", authorization="Bearer sk-xyz987654321",
                 nested={"token": "gho_ABCDEFGHIJKLMNOP", "note": "key sk-abcdef1234567890 leaked"},
                 prompt="正常内容")
        r = self._lines()[0]
        self.assertEqual(r["api_key"], "***")
        self.assertEqual(r["authorization"], "***")
        self.assertEqual(r["nested"]["token"], "***")
        self.assertNotIn("sk-abcdef1234567890", json.dumps(r, ensure_ascii=False))
        self.assertNotIn("gho_", json.dumps(r, ensure_ascii=False))
        self.assertIn("正常内容", r["prompt"])

    def test_redact_pure(self):
        self.assertEqual(redact({"api_key": "x", "n": 1}), {"api_key": "***", "n": 1})
        self.assertEqual(redact("token sk-1234567890")[0:5], "token")
        self.assertNotIn("sk-1234567890", redact("token sk-1234567890"))
        self.assertEqual(redact(["sk-1234567890"]), ["***"])
        clipped = redact("y" * 3000, clip=100)
        self.assertLess(len(clipped), 200)
        self.assertIn("+2900字", clipped)

    def test_bind_context(self):
        log = get_logger().bind(run_id="r1", session="s1")
        log.info("inside")
        r = self._lines()[0]
        self.assertEqual((r["run_id"], r["session"]), ("r1", "s1"))
        self.assertEqual(log.files(), get_logger().files())   # 子日志器与父同一落盘目标

    def test_read_tail_and_errors(self):
        log = get_logger()
        log.info("a")
        log.warning("w")
        log.error("e")
        self.assertEqual([r["event"] for r in log.read(limit=2)], ["w", "e"])
        self.assertEqual([r["event"] for r in log.errors()], ["w", "e"])
        self.assertEqual([r["event"] for r in log.read(level="ERROR")], ["e"])

    def test_stats_and_clear(self):
        log = get_logger()
        log.info("x")
        st = log.stats()
        self.assertEqual(st["files"], 1)
        self.assertGreater(st["bytes"], 0)
        self.assertEqual(log.clear(), 1)
        self.assertEqual(log.files(), [])

    def test_unwritable_dir_never_raises(self):
        bad = os.path.join(self.tmp, "no", "such", "deep")
        with patch("os.makedirs", side_effect=PermissionError("nope")):
            log = Logger(log_dir=bad)
        self.assertFalse(log.enabled)
        self.assertIn("不可用", log.last_error or "")
        log.info("不会抛")                                  # 关键：日志挂了不拖垮主流程
        log_event("ERROR", "也不抛")

    def test_singleton_and_log_event(self):
        reset_logger()
        log = get_logger({"logging": {"dir": self.tmp, "level": "INFO"}})
        self.assertIs(log, get_logger())
        log_event("WARNING", "via_helper", k=1)
        self.assertEqual(self._lines()[-1]["event"], "via_helper")


try:
    from fastapi.testclient import TestClient
except Exception:                                              # pragma: no cover
    TestClient = None


@unittest.skipUnless(TestClient is not None, "未装服务依赖（pip install handcraft-agent[server]）")
class TestServerLogIntegration(unittest.TestCase):
    def test_http_request_logged(self):
        from forge.server import ForgeService, SessionStore, create_app

        tmp = tempfile.mkdtemp(prefix="forge-httplog-")
        init_logger({"logging": {"dir": tmp, "level": "INFO", "enabled": True}})
        self.addCleanup(reset_logger)
        svc = ForgeService(cfg={"server": {"db_path": os.path.join(tmp, "s.db")}},
                           store=SessionStore(os.path.join(tmp, "s.db")),
                           agent_factory=lambda sid: None)
        app = create_app(cfg={"server": {}}, service=svc, api_keys=["k"], rate_limit=0)
        c = TestClient(app)
        self.assertEqual(c.get("/healthz").status_code, 200)
        self.assertEqual(c.get("/api/status", headers={"X-API-Key": "k"}).status_code, 200)
        rows = [r for r in get_logger().read(limit=20) if r["event"] == "http_request"]
        self.assertTrue(rows)
        self.assertEqual(rows[-1]["path"], "/api/status")
        self.assertEqual(rows[-1]["status"], 200)
        self.assertEqual(rows[-1]["client"], "testclient")


if __name__ == "__main__":
    unittest.main(verbosity=2)
