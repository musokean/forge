"""#12 Web 界面测试：零依赖 HTTP 服务端到端验证。

运行：python test_web.py
依赖：managed venv（不联网——Agent 用 mock，起本地 127.0.0.1 随机端口真实服务）
"""
import json
import sys
import threading
import time
import unittest
import urllib.request
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, ".")

from src.web import ForgeWeb, _PAGE_HTML


def _fake_agent(reply="测试回复"):
    a = SimpleNamespace(
        run=AsyncMock(return_value=reply),
        reset=lambda: None,
        total_tokens={"prompt": 1, "completion": 1},
        messages=[],
    )
    return a


class WebServerBase(unittest.TestCase):
    """共享：起一个真实 HTTP 服务（mock Agent），测完关闭。"""

    @classmethod
    def setUpClass(cls):
        cls.agent = _fake_agent()
        cls.web = ForgeWeb(host="127.0.0.1", port=0, agent=cls.agent, auto_open=False)
        cls.url = cls.web.start()
        cls.base = cls.url.rstrip("/")

    @classmethod
    def tearDownClass(cls):
        cls.web.stop()

    def _get(self, path):
        req = urllib.request.Request(self.base + path, headers={"Connection": "close"})
        opener = urllib.request.build_opener()  # 新连接池，杜绝 keep-alive 复用竞态（Windows 10053）
        with opener.open(req, timeout=5) as r:
            return r.status, r.read().decode("utf-8")

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json", "Connection": "close"})
        opener = urllib.request.build_opener()  # 新连接池，杜绝 keep-alive 复用竞态（Windows 10053）
        with opener.open(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode("utf-8"))


class TestPage(WebServerBase):
    def test_index_returns_html(self):
        status, html = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("FORGE", html)
        self.assertIn("把想法锻造成现实", html)
        self.assertIn("api/chat", html)   # 前端调用的接口在页面里

    def test_index_html_is_self_contained(self):
        """页面必须零外部资源（内嵌 CSS/JS，无外链）——本机离线可用。"""
        self.assertNotIn("http://", _PAGE_HTML.replace("http://127.0.0.1", ""))
        self.assertNotIn("https://", _PAGE_HTML)
        self.assertIn("<style>", _PAGE_HTML)
        self.assertIn("<script>", _PAGE_HTML)


class TestChatAPI(WebServerBase):
    def test_chat_returns_reply(self):
        status, d = self._post("/api/chat", {"message": "你好"})
        self.assertEqual(status, 200)
        self.assertEqual(d["reply"], "测试回复")

    def test_chat_empty_message_400(self):
        try:
            self._post("/api/chat", {"message": "  "})
            self.fail("应返回 400")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_chat_invokes_agent_once(self):
        self.agent.run.reset_mock()
        self._post("/api/chat", {"message": "算一下 1+1"})
        self.agent.run.assert_awaited_once()

    def test_reset(self):
        status, d = self._post("/api/reset", {})
        self.assertEqual(status, 200)
        self.assertTrue(d["ok"])

    def test_write_rejection_warning(self):
        """模型回答里出现写操作被拒 → 返回一次性提示。"""
        self.agent.run = AsyncMock(return_value="抱歉，用户拒绝了该写操作（write_file）")
        self.web._warned.clear()
        status, d = self._post("/api/chat", {"message": "帮我写个文件"})
        self.assertEqual(status, 200)
        self.assertIn("Web 端被安全默认拒绝", d.get("warning", ""))


class TestStatusAPI(WebServerBase):
    def test_status_returns_model(self):
        status, html = self._get("/api/status")
        self.assertEqual(status, 200)
        d = json.loads(html)
        self.assertIn("model", d)
        self.assertEqual(d["host"], "127.0.0.1")


class TestWebLifecycle(unittest.TestCase):
    def test_start_returns_url_and_thread(self):
        web = ForgeWeb(host="127.0.0.1", port=0, agent=_fake_agent(), auto_open=False)
        url = web.start()
        self.assertTrue(url.startswith("http://127.0.0.1:"))
        self.assertIsNotNone(web._thread)
        self.assertTrue(web._thread.is_alive())
        # 能请求
        with urllib.request.urlopen(url + "/api/status", timeout=5) as r:
            self.assertEqual(r.status, 200)
        web.stop()
        self.assertIsNone(web._server)

    def test_memory_hook_invoked(self):
        """/api/chat 会走长期记忆钩子（auto_remember + compose_context）。"""
        agent = _fake_agent()
        web = ForgeWeb(host="127.0.0.1", port=0, agent=agent, auto_open=False)
        web.start()
        try:
            mem = SimpleNamespace(auto_remember=lambda s: None, compose_context=lambda s: "")
            with patch("src.memory.get_memory", return_value=mem):
                data = json.dumps({"message": "你好"}).encode("utf-8")
                req = urllib.request.Request(web.url.rstrip("/") + "/api/chat", data=data,
                                             headers={"Content-Type": "application/json", "Connection": "close"})
                opener = urllib.request.build_opener()
                with opener.open(req, timeout=15) as r:
                    d = json.loads(r.read().decode("utf-8"))
            self.assertEqual(d["reply"], "测试回复")
        finally:
            web.stop()


class TestAgentDefaults(unittest.TestCase):
    def test_web_agent_auto_reject(self):
        """Web 端 Agent 默认写操作全拒绝（安全默认，无交互审批通道）。"""
        web = ForgeWeb(host="127.0.0.1", port=0, auto_open=False)  # 不传 agent → 新建
        try:
            self.assertEqual(web.agent.approver.mode, "auto_reject")
        finally:
            web.stop()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
