"""CLI / REPL 命令层测试（离线）。

重点守一类**静默 bug**：REPL 的命令分派链里每条命令处理完必须 `continue`，
少一个就会「命令执行了 + 同一句话又被丢给模型」（重复执行 + 白烧 token）。
2026-09-28 实测踩到（/web、/serve、/logs、/device 四条漏了 continue）。

另外覆盖新增命令在 CLI 层不炸：/device、/logs、/sandbox（sim 模式与写配置路径）。
"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, ".")

import main as cli                                     # noqa: E402
from forge.logging_setup import init_logger, reset_logger  # noqa: E402


def _run(fn, *a):
    """跑一个 CLI 命令函数，返回它打印的内容。"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*a)
    return buf.getvalue()


class TestReplDispatch(unittest.TestCase):
    def test_every_command_branch_continues(self):
        """分派链里每个 `_xxx_command(...)` 后面必须紧跟 continue。"""
        src = open("main.py", encoding="utf-8").read().split("\n")
        # 只在 _repl 的分派区里检查（从 `async def _repl` 到 `def main(`）
        start = next(i for i, l in enumerate(src) if l.startswith("async def _repl"))
        end = next(i for i, l in enumerate(src) if l.startswith("def main(") and i > start)
        missing = []
        for i in range(start, end):
            s = src[i].strip()
            if s.startswith("_") and "_command(" in s and s.endswith(")"):
                nxt = [x.strip() for x in src[i + 1:i + 3] if x.strip()]
                if not nxt or nxt[0] != "continue":
                    missing.append((i + 1, s))
        self.assertEqual(missing, [], f"这些命令分支缺 continue：{missing}")

    def test_dispatch_contains_new_commands(self):
        src = open("main.py", encoding="utf-8").read()
        for cmd in ("/device", "/sandbox", "/logs", "/serve", "/web"):
            self.assertIn(f'line == "{cmd}"', src)

    def test_help_lists_new_commands(self):
        text = cli._help() if hasattr(cli, "_help") else ""
        if not text:
            src = open("main.py", encoding="utf-8").read()
            text = src
        for cmd in ("/device", "/sandbox", "/logs"):
            self.assertIn(cmd, text)


class TestCliCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge-cli-")
        init_logger({"logging": {"dir": self.tmp, "level": "INFO", "enabled": True}})
        self.addCleanup(reset_logger)

    def test_device_command_in_sim_mode(self):
        """Phase 0（默认 sim）：/device 打印模拟器状态，不报错。"""
        cfg = {"device": {"enabled": False, "transport": "sim"}}
        with patch("forge.config.load_config", return_value=cfg):
            out = _run(cli._device_command, "")
        self.assertIn("Phase 0", out)
        self.assertIn("temperature_c", out)

    def test_device_command_usage_hint(self):
        cfg = {"device": {"enabled": False, "transport": "sim"}}
        with patch("forge.config.load_config", return_value=cfg):
            out = _run(cli._device_command, "乱输入")
        self.assertIn("用法", out)

    def test_device_mode_writes_config(self):
        """`/device mode serial <url>` 要写回配置（老配置缺段时自愈补段）。"""
        import forge.config_writer as cw

        path = os.path.join(self.tmp, "models.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("models: {}\n")
        with patch.object(cw, "config_path", return_value=path):
            out = _run(cli._device_command, "mode serial socket://127.0.0.1:9009")
        self.assertIn("serial", out)
        text = open(path, encoding="utf-8").read()
        self.assertIn("device:", text)
        self.assertIn("transport: serial", text)
        self.assertIn('serial_url: "socket://127.0.0.1:9009"', text)

    def test_logs_command_status_and_tail(self):
        from forge.logging_setup import get_logger

        get_logger().info("cli_test_event", k=1)
        out = _run(cli._logs_command, "")
        self.assertIn("日志", out)
        out2 = _run(cli._logs_command, "tail 5")
        self.assertIn("cli_test_event", out2)
        out3 = _run(cli._logs_command, "path")
        self.assertIn(self.tmp, out3)

    def test_sandbox_command_status_and_test(self):
        with patch("forge.config.load_config", return_value={"sandbox": {"mode": "local"}}):
            out = _run(cli._sandbox_command, "")
            self.assertIn("沙箱", out)
            out2 = _run(cli._sandbox_command, "test echo cli-sandbox-ok")
        self.assertIn("cli-sandbox-ok", out2)

    def test_sandbox_dangerous_command_blocked_via_cli(self):
        with patch("forge.config.load_config", return_value={"sandbox": {"mode": "local"}}):
            out = _run(cli._sandbox_command, "test rm -rf /")
        self.assertIn("⛔", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
