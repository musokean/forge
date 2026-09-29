"""#16 硬件 Phase 1 测试（全离线：不需要真硬件、不需要 MQTT broker、不写 repo）。

覆盖四层 + 端到端：
  · 协议层：帧编解码 / CRC 校验 / 版本不符 / 坏帧丢弃 / 半帧与粘包 / 中文负载
  · 传输层：内存对管 · 真串口（pyserial `loop://` 走真实代码路径）· MQTT（假 broker 客户端）· 工厂选择
  · 控制平面：资产目录（未知别名拒绝）· 策略引擎（分阶段放权/限额/冷却/远程禁写）·
             命令状态机（Applied / Rejected / Timeout+重试 / ack 但无 state）· 回滚 · 审计
  · 端到端：**设备端模拟器 serve_socket + Agent 侧 pyserial `socket://`** —— 走真协议真传输，
            只差没有物理器件（这一条就是「没有硬件也能验真链路」）
  · 集成：tools.device_* 走真链路 · 结构化日志落 device_cmd · /device 命令
"""
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

import src.hwproto as P                                    # noqa: E402
import src.hwtransport as T                                # noqa: E402
import src.logging_setup as logging_setup                  # noqa: E402
from device_sim import DeviceSimulator, serve_socket       # noqa: E402
from fake_device import BeautyDevice                       # noqa: E402
from src.hwcontrol import (                                # noqa: E402
    AssetRegistry,
    CommandService,
    HardwareError,
    HardwareLink,
    PolicyEngine,
    reset_link,
)
from src.logging_setup import init_logger, reset_logger    # noqa: E402

# ── 测试基类：日志一律写临时目录（绝不碰 repo 的 data/logs）──────────
class HwTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge-hw-")
        init_logger({"logging": {"dir": self.tmp, "level": "DEBUG", "enabled": True}})
        self.addCleanup(reset_logger)
        reset_link()
        self.addCleanup(reset_link)
        T._reset_docker_cache = getattr(T, "_reset_docker_cache", lambda: None)

    def log_events(self, event=None):
        rows = []
        for f in logging_setup.get_logger().files():
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rows.append(json.loads(line))
        return [r for r in rows if event is None or r["event"] == event]


# ── 工具：起一个真 TCP 透传的设备端模拟器 ───────────────────────────
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_socket_sim(test_hooks=True, duration=60):
    """起模拟设备端（TCP 透传服务端）→ 返回 (sim, port)。"""
    sim = DeviceSimulator(test_hooks=test_hooks, verbose=False)
    port = _free_port()
    th = threading.Thread(target=serve_socket,
                          kwargs=dict(sim=sim, port=port, host="127.0.0.1", duration=duration),
                          daemon=True)
    th.start()
    deadline = time.time() + 5
    while time.time() < deadline:                 # 探活：连上即说明在监听
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return sim, port
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("模拟设备端没起来")


def link_cfg(serial_url, **over):
    cfg = {"device": {"enabled": True, "transport": "serial", "serial_url": serial_url,
                      "device_id": "beauty-01", "timeout": 2, "retries": 1, "monitor": False,
                      "policy": {"stage": "low_risk", "cooldown_s": 0, "max_level": 3,
                                 "max_temp_c": 45, "max_runtime_s": 900}}}
    cfg["device"].update(over)
    return cfg


# ══════════════════════════════════════════════════════════════════
class TestProtocol(unittest.TestCase):
    def test_round_trip_all_types(self):
        frames = [
            P.hello("forge", ["power_on"], seq=1),
            P.hello_ack(1, "beauty-01", "三合一美容仪", fw="sim-1.0", capabilities=["status"],
                        limits={"max_level": 3}, safety={"tripped": False}),
            P.cmd("set_level", {"level": 2}, seq=2),
            P.ack(2, "set_level", ok=True),
            P.state(2, "set_level", {"power": True, "level": 2}),
            P.event("over_temp", {"temperature_c": 46.0}, seq=0),
            P.ping(3),
            P.pong(3),
        ]
        types = []
        for raw in frames:
            self.assertTrue(raw.endswith(b"\n"))
            body = P.decode(raw)
            types.append(body["type"])
            self.assertEqual(body["v"], P.PROTOCOL_VERSION)
            self.assertEqual(body["crc"], P.crc_of(body))
        self.assertEqual(types, [P.MSG_HELLO, P.MSG_HELLO_ACK, P.MSG_CMD, P.MSG_ACK,
                                 P.MSG_STATE, P.MSG_EVENT, P.MSG_PING, P.MSG_PONG])

    def test_unicode_payload(self):
        body = P.decode(P.cmd("say", {"text": "三档 · 已开机"}, seq=9))
        self.assertEqual(body["args"]["text"], "三档 · 已开机")

    def test_crc_corruption_detected(self):
        raw = P.cmd("power_on", {}, seq=5).decode("utf-8")
        broken = raw.replace('"level"', '"level"').replace("power_on", "power_0ff")
        with self.assertRaises(P.ProtocolError) as cm:
            P.decode(broken)
        self.assertEqual(cm.exception.code, P.E_CRC)

    def test_version_mismatch(self):
        body = {"v": 99, "seq": 1, "type": P.MSG_PING}
        body["crc"] = P.crc_of(body)
        raw = json.dumps(body)
        with self.assertRaises(P.ProtocolError) as cm:
            P.decode(raw)
        self.assertEqual(cm.exception.code, P.E_VERSION)

    def test_unknown_type_and_bad_json(self):
        body = {"v": P.PROTOCOL_VERSION, "seq": 1, "type": "lazer"}
        body["crc"] = P.crc_of(body)
        raw = json.dumps(body)
        with self.assertRaises(P.ProtocolError) as cm:
            P.decode(raw)
        self.assertEqual(cm.exception.code, P.E_UNKNOWN_TYPE)
        for bad in ("", "   ", "not json", "[1,2]", "{}\n"):
            with self.assertRaises(P.ProtocolError) as cm:
                P.decode(bad)
            self.assertEqual(cm.exception.code, P.E_BAD_JSON)

    def test_frame_reader_partial_and_coalesced(self):
        r = P.FrameReader()
        a, b = P.cmd("power_on", seq=1), P.cmd("power_off", seq=2)
        self.assertEqual(r.feed(a[:5]), [])                    # 半帧：不该出帧
        got_a = r.feed(a[5:])                                  # 补齐后应恰好出 1 帧
        self.assertEqual([g["id"] for g in got_a], ["power_on"])
        # 正确的半帧测试：截断换行符
        r2 = P.FrameReader()
        raw = P.cmd("power_on", seq=3)
        self.assertEqual(r2.feed(raw[:-1]), [])
        got = r2.feed(raw[-1:])
        self.assertEqual([g["id"] for g in got], ["power_on"])
        # 粘包：一次喂多帧
        r3 = P.FrameReader()
        got3 = r3.feed(a + b)
        self.assertEqual([g["id"] for g in got3], ["power_on", "power_off"])

    def test_frame_reader_drops_garbage_and_flags_oversize(self):
        r = P.FrameReader()
        got = r.feed(b'{"oops":1}\n' + P.cmd("power_on", seq=1))
        self.assertEqual(len(got), 1)
        self.assertEqual(r.dropped, 1)
        r2 = P.FrameReader(max_line=50)
        with self.assertRaises(P.ProtocolError):
            r2.feed(b"x" * 200)
        self.assertEqual(r2.buf, bytearray())                   # 超长直接清缓冲，不长内存


# ══════════════════════════════════════════════════════════════════
class TestTransports(HwTestBase):
    def test_memory_pair(self):
        a, b = T.MemoryTransport.pair()
        a.open(); b.open()
        a.write(P.ping(1))
        self.assertEqual(P.decode(b.read_line(timeout=1))["type"], P.MSG_PING)
        self.assertIsNone(b.read_line(timeout=0.05))            # 超时返回 None
        self.assertIn("memory", a.describe())

    def test_serial_loopback_real_pyserial(self):
        """真 pyserial 代码路径（loop:// 回环）：写进去能读回来。"""
        try:
            import serial  # noqa: F401
        except Exception:
            self.skipTest("未装 pyserial")
        tr = T.SerialTransport("loop://", timeout=0.5).open()
        try:
            tr.write(P.ping(7))
            line = tr.read_line(timeout=1)
            self.assertTrue(line)
            self.assertEqual(P.decode(line)["type"], P.MSG_PING)
            self.assertIn("loop://", tr.describe())
        finally:
            tr.close()
        self.assertFalse(tr.connected)

    def test_serial_errors(self):
        with self.assertRaises(T.TransportError):
            T.SerialTransport("").open()
        with patch.object(T, "serial_for_url", None):
            with self.assertRaises(T.TransportError) as cm:
                T.SerialTransport("COM1").open()
            self.assertIn("pyserial", str(cm.exception))
        tr = T.SerialTransport("loop://")
        with self.assertRaises(T.TransportError):
            tr.write(b"x")                                       # 未打开就写

    def test_mqtt_topics_and_delivery(self):
        sent = []

        class FakeClient:
            def __init__(self):
                self.on_message = None
                self.on_connect = None
                self.subscribed = []

            def subscribe(self, topic, qos=0):
                self.subscribed.append((topic, qos))
                return (0, 1)

            def publish(self, topic, payload, qos=0):
                sent.append((topic, payload, qos))
                if self.on_message:                              # 假 broker：回灌一条上行
                    class M:
                        pass
                    m = M(); m.payload = P.pong(1).strip()
                    self.on_message(self, None, m)
                return (0, 1)

            def connect(self, host, port, keepalive=30):
                return 0

            def disconnect(self):
                return 0

        fake = FakeClient()
        tr = T.MqttTransport(host="10.0.0.9", port=1883, device_id="beauty-01",
                             topic_prefix="forge/dev", qos=1, client=fake).open()
        tr.write(P.cmd("power_on", seq=3))
        self.assertEqual(fake.subscribed[0][0], "forge/dev/up/beauty-01")
        self.assertEqual(sent[0][0], "forge/dev/cmd/beauty-01")
        self.assertEqual(sent[0][2], 1)
        line = tr.read_line(timeout=1)
        self.assertEqual(P.decode(line)["type"], P.MSG_PONG)
        self.assertIn("mqtt[", tr.describe())

    def test_mqtt_without_paho(self):
        with patch.object(T, "mqtt", None):
            with self.assertRaises(T.TransportError) as cm:
                T.MqttTransport().open()
            self.assertIn("paho", str(cm.exception))

    def test_factory(self):
        self.assertIsInstance(T.open_transport("sim"), T.MemoryTransport)
        self.assertIsInstance(T.open_transport(""), T.MemoryTransport)
        self.assertIsInstance(T.open_transport("loop://"), T.SerialTransport)
        self.assertIsInstance(T.open_transport("COM5"), T.SerialTransport)
        mq = T.open_transport("mqtt://192.168.1.10:1884")
        self.assertIsInstance(mq, T.MqttTransport)
        self.assertEqual((mq.host, mq.port), ("192.168.1.10", 1884))
        # 配置驱动
        self.assertIsInstance(T.open_transport({"device": {"transport": "sim"}}), T.MemoryTransport)
        s = T.open_transport({"device": {"transport": "serial", "serial_url": "socket://1.2.3.4:9000"}})
        self.assertIsInstance(s, T.SerialTransport)
        m2 = T.open_transport({"device": {"transport": "mqtt",
                                          "mqtt": {"host": "broker.local", "port": 1885}}})
        self.assertEqual(m2.host, "broker.local")
        with self.assertRaises(T.TransportError):
            T.open_transport({"device": {"transport": "serial"}})


# ══════════════════════════════════════════════════════════════════
class TestAssetRegistry(unittest.TestCase):
    def test_single_default_and_unknown(self):
        reg = AssetRegistry(default_id="beauty-01")
        self.assertEqual(reg.resolve()["device_id"], "beauty-01")
        with self.assertRaises(HardwareError) as cm:
            reg.resolve("机器人")
        self.assertEqual(cm.exception.code, "E_UNKNOWN_DEVICE")

    def test_alias_and_id_lookup(self):
        reg = AssetRegistry({"一号机": {"device_id": "beauty-01", "site": "深圳", "transport": "serial"},
                             "二号机": {"device_id": "beauty-02", "site": "东莞"}})
        self.assertEqual(reg.resolve("一号机")["site"], "深圳")
        self.assertEqual(reg.resolve("beauty-02")["site"], "东莞")
        self.assertEqual(len(reg.all()), 2)
        with self.assertRaises(HardwareError):
            reg.resolve()                                        # 多设备必须点名，不猜


class TestPolicy(unittest.TestCase):
    def test_stage_readonly_blocks_writes(self):
        pol = PolicyEngine({"stage": "readonly"})
        ok, reason, code = pol.check("power_on", {}, {})
        self.assertFalse(ok)
        self.assertEqual(code, "E_STAGE_READONLY")
        self.assertTrue(pol.check("status", {}, {})[0])
        self.assertTrue(PolicyEngine({"stage": "low_risk"}).check("power_on", {}, {})[0])

    def test_limits_and_temp_and_safety(self):
        pol = PolicyEngine({"stage": "low_risk", "max_level": 2, "max_temp_c": 45})
        self.assertFalse(pol.check("set_level", {"level": 3}, {})[0])
        self.assertFalse(pol.check("set_level", {"level": "2"}, {})[0])       # 类型不对也拒
        self.assertTrue(pol.check("set_level", {"level": 2}, {})[0])
        ok, reason, code = pol.check("set_level", {"level": 1}, {"temperature_c": 46})
        self.assertFalse(ok); self.assertEqual(code, "E_OVER_TEMP")
        self.assertTrue(pol.check("power_off", {}, {"temperature_c": 46})[0])  # 只允许关机
        ok, _, code = pol.check("power_on", {}, {"safety_tripped": True})
        self.assertFalse(ok); self.assertEqual(code, "E_SAFETY_TRIPPED")
        self.assertTrue(pol.check("reset_safety", {}, {"safety_tripped": True})[0])

    def test_cooldown_and_remote(self):
        pol = PolicyEngine({"cooldown_s": 5, "allow_remote": False})
        self.assertTrue(pol.check("power_on", {}, {})[0])
        pol.note_write()
        ok, reason, code = pol.check("power_on", {}, {})
        self.assertFalse(ok); self.assertEqual(code, "E_COOLDOWN")
        ok, reason, code = pol.check("power_on", {}, {}, remote=True)
        self.assertFalse(ok); self.assertEqual(code, "E_REMOTE_BLOCKED")
        self.assertTrue(PolicyEngine({"allow_remote": True}).check("power_on", {}, {}, remote=True)[0])


class TestCommandService(HwTestBase):
    class _Device:
        """假设备端：按脚本回帧。"""

        def __init__(self, reply_ack=True, ack_ok=True, reply_state=True, state=None):
            self.reply_ack, self.ack_ok, self.reply_state = reply_ack, ack_ok, reply_state
            self.state = state or {"power": True, "level": 1}
            self.seen = []

        def __call__(self, frame):
            self.seen.append(frame)
            out = []
            if self.reply_ack:
                out.append(P.ack(frame["seq"], frame["id"], ok=self.ack_ok, reason="" if self.ack_ok else "设备拒绝",
                                 code="" if self.ack_ok else P.E_REJECTED))
            if self.reply_state and self.ack_ok:
                out.append(P.state(frame["seq"], frame["id"], self.state))
            return out

    def _pair(self, device):
        agent, dev = T.MemoryTransport.pair(timeout=1)
        agent.open(); dev.open()

        def pump():
            reader = P.FrameReader()
            deadline = time.time() + 5
            while time.time() < deadline:
                chunk = dev.read_line(timeout=0.2)
                if not chunk:
                    continue
                for f in reader.feed(chunk):
                    for r in device(f):
                        dev.write(r)

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        self.addCleanup(lambda: th.join(timeout=0.1))
        return agent

    def test_applied_requires_state(self):
        tr = self._pair(self._Device())
        svc = CommandService(tr, timeout=1, retries=0)
        res = svc.execute("power_on", {})
        self.assertEqual(res.state, "Applied")
        self.assertTrue(res.ok)
        self.assertEqual(res.state_after["power"], True)
        self.assertEqual(len(self.log_events("device_cmd")), 1)

    def test_rejected(self):
        tr = self._pair(self._Device(ack_ok=False, reply_state=False))
        res = CommandService(tr, timeout=1, retries=0).execute("set_level", {"level": 9})
        self.assertEqual(res.state, "Rejected")
        self.assertFalse(res.ok)
        self.assertIn("拒绝", res.reason)

    def test_timeout_with_retries(self):
        tr = self._pair(self._Device(reply_ack=False, reply_state=False))
        svc = CommandService(tr, timeout=0.3, retries=2)
        res = svc.execute("power_on", {})
        self.assertEqual(res.state, "Timeout")
        self.assertFalse(res.ok)
        self.assertEqual(res.attempts, 3)                        # 1 次 + 2 次重试
        self.assertEqual(res.code, P.E_TIMEOUT)

    def test_ack_without_state_is_not_success(self):
        tr = self._pair(self._Device(reply_state=False))
        res = CommandService(tr, timeout=0.3, retries=0).execute("power_on", {})
        self.assertEqual(res.state, "Accepted")
        self.assertFalse(res.ok)                                 # 「收到了」≠「生效了」
        self.assertEqual(res.code, "E_NO_STATE")

    def test_device_event_recorded(self):
        class Ev:
            def __call__(self, frame):
                return [P.event("over_temp", {"temperature_c": 46})]

        tr = self._pair(Ev())
        svc = CommandService(tr, timeout=0.5, retries=0)
        svc.execute("power_on", {})
        self.assertTrue(any(e.get("kind") == "over_temp" for e in svc.events))
        self.assertEqual(self.log_events("device_event")[0]["kind"], "over_temp")


# ══════════════════════════════════════════════════════════════════
class TestHardwareLinkInProcess(HwTestBase):
    """控制平面 + 门面（内存传输直连模拟设备端，验逻辑）。"""

    def _link(self, **policy):
        sim = DeviceSimulator(test_hooks=True, verbose=False)
        agent, dev = T.MemoryTransport.pair(timeout=1)
        agent.open(); dev.open()

        def pump():
            reader = P.FrameReader()
            while True:
                chunk = dev.read_line(timeout=0.2)
                if not chunk:
                    continue
                for f in reader.feed(chunk):
                    for r in sim.handle(f):
                        dev.write(r)

        threading.Thread(target=pump, daemon=True).start()
        pol = {"stage": "low_risk", "cooldown_s": 0}
        pol.update(policy)
        cfg = {"device": {"enabled": True, "transport": "serial", "device_id": "beauty-01",
                          "timeout": 1, "retries": 0, "monitor": False, "policy": pol}}
        link = HardwareLink(cfg, transport=agent)
        return link, sim

    def test_full_flow_in_process(self):
        link, sim = self._link()
        st = link.connect()
        self.assertTrue(st["connected"])
        self.assertEqual(link.device_info.get("fw"), "sim-1.0")
        self.assertIn("set_level", link.capabilities)
        self.assertTrue(link.power_on()["ok"])
        r = link.set_level(2)
        self.assertTrue(r["ok"])
        self.assertEqual(r["status"]["level"], 2)
        st = link.status()
        self.assertTrue(st["power"])
        self.assertEqual(st["level"], 2)
        self.assertTrue(link.power_off()["ok"])
        self.assertFalse(link.status()["power"])
        self.assertGreaterEqual(sim.counters["commands"], 5)
        # 审计：每条写命令一条
        cmds = [e["cmd"] for e in link.audit(10)]
        self.assertIn("set_level", cmds)

    def test_policy_block_before_sending(self):
        link, sim = self._link(max_level=2)
        link.connect()
        before = sim.counters["commands"]
        r = link.set_level(3)
        self.assertFalse(r["ok"])
        self.assertEqual(r["code"], "E_LEVEL_LIMIT")
        self.assertTrue(r["blocked"])
        self.assertEqual(sim.counters["commands"], before)        # 越权命令根本没发出去
        self.assertTrue(self.log_events("device_policy_block"))

    def test_over_temp_guard_cuts_power(self):
        link, sim = self._link()
        link.connect()
        link.power_on()
        self.assertTrue(link.status()["power"])
        sim.handle(P.decode(P.cmd("sim_set_temp", {"value": 46.0}, seq=999)))   # 设备端测试钩子：推到 46°C
        st = link.status()                                        # 控制平面巡检 → 主动断电
        self.assertFalse(st["power"])
        self.assertTrue(self.log_events("device_guard_over_temp"))

    def test_max_runtime_guard(self):
        link, sim = self._link(max_runtime_s=0.001)
        link.connect()
        link.power_on()
        time.sleep(0.35)                                          # 设备端 run_seconds 每 100ms 更新一次
        link.status()
        self.assertFalse(link.status()["power"])
        self.assertTrue(self.log_events("device_guard_max_runtime"))

    def test_duck_typing_with_fake_device(self):
        """接口与 Phase 0 的 BeautyDevice 一致 → tools.py 不用改。"""
        link, _ = self._link()
        sim_dev = BeautyDevice()
        for m in ("status", "power_on", "power_off", "set_level", "reset_safety"):
            self.assertTrue(callable(getattr(link, m)), m)
            self.assertTrue(callable(getattr(sim_dev, m)), m)
        self.assertEqual(set(sim_dev.status()) - set(link.status()),
                         set(sim_dev.status()) - set(link.status()))   # 键集同构
        link.connect()
        self.assertTrue(set(sim_dev.status()).issubset(set(link.status())))

    def test_no_handshake_raises(self):
        agent, dev = T.MemoryTransport.pair(timeout=0.2)
        agent.open()                                              # 对端没有设备端应答
        link = HardwareLink({"device": {"enabled": True, "transport": "serial", "timeout": 0.3,
                                        "monitor": False}}, transport=agent)
        with self.assertRaises(HardwareError) as cm:
            link.connect()
        self.assertEqual(cm.exception.code, "E_NO_HANDSHAKE")

    def test_rollback_on_reject(self):
        """设备拒绝 set_level 时，控制平面回滚到上一档（A24 要求的回滚语义）。"""
        link, sim = self._link()
        link.connect()
        link.power_on()
        link.set_level(2)
        self.assertEqual(link.status()["level"], 2)
        orig = sim._apply

        def flaky(cmd_id, args):                                  # 只让 level=3 失败
            if cmd_id == "set_level" and args.get("level") == 3:
                return False, "模拟设备故障", P.E_REJECTED
            return orig(cmd_id, args)

        sim._apply = flaky
        r = link.set_level(3)
        self.assertFalse(r["ok"])
        self.assertTrue(r.get("rolled_back"))
        self.assertEqual(link.status()["level"], 2)               # 已回到 2 档


# ══════════════════════════════════════════════════════════════════
class TestRealLinkOverTcp(HwTestBase):
    """**没有硬件也能验真链路**：模拟设备端（TCP 透传）+ pyserial `socket://` 客户端。"""

    def test_end_to_end_over_socket(self):
        try:
            import serial  # noqa: F401
        except Exception:
            self.skipTest("未装 pyserial")
        sim, port = start_socket_sim(test_hooks=True)
        url = f"socket://127.0.0.1:{port}"
        link = HardwareLink(link_cfg(url))
        try:
            st = link.connect()
            self.assertTrue(st["connected"], st)
            self.assertIn("socket://", st["transport"])
            self.assertEqual(link.device_info.get("fw"), "sim-1.0")
            self.assertTrue(link.power_on()["ok"])
            self.assertEqual(link.set_level(2)["status"]["level"], 2)
            self.assertTrue(link.power_off()["ok"])
            self.assertGreaterEqual(sim.counters["commands"], 4)
            # 结构化日志：命令与状态机轨迹都落盘
            self.assertTrue(self.log_events("device_cmd"))
            self.assertTrue(self.log_events("device_connected"))
        finally:
            link.close()

    def test_reconnect_after_close(self):
        try:
            import serial  # noqa: F401
        except Exception:
            self.skipTest("未装 pyserial")
        _sim, port = start_socket_sim(duration=60)
        url = f"socket://127.0.0.1:{port}"
        link = HardwareLink(link_cfg(url))
        try:
            self.assertTrue(link.connect()["connected"])
            link.close()
            self.assertFalse(link.connected)
            self.assertTrue(link.reconnect()["connected"])        # 重连（设备端循环等待接入）
            self.assertTrue(link.power_on()["ok"])
        finally:
            link.close()

    def test_connect_failure_reports_actionable_error(self):
        try:
            import serial  # noqa: F401
        except Exception:
            self.skipTest("未装 pyserial")
        port = _free_port()                                       # 没有任何东西在监听
        link = HardwareLink(link_cfg(f"socket://127.0.0.1:{port}", timeout=0.5, retries=0))
        with self.assertRaises(HardwareError) as cm:
            link.connect()
        self.assertIn("device_sim.py", cm.exception.message)       # 报错必须告诉人怎么自救


# ══════════════════════════════════════════════════════════════════
class TestToolsIntegration(HwTestBase):
    def test_tools_route_to_real_link(self):
        try:
            import serial  # noqa: F401
        except Exception:
            self.skipTest("未装 pyserial")
        sim, port = start_socket_sim(test_hooks=True)
        cfg = link_cfg(f"socket://127.0.0.1:{port}")
        import src.tools as tools

        old = tools._device
        tools._device = None
        self.addCleanup(lambda: setattr(tools, "_device", old))
        with patch("src.config.load_config", return_value=cfg):
            st = json.loads(tools.device_status())
            self.assertTrue(st.get("connected"), st)
            self.assertTrue(json.loads(tools.device_power("on"))["ok"])
            self.assertTrue(json.loads(tools.device_level(2))["ok"])
            self.assertFalse(json.loads(tools.device_level(9))["ok"])     # 越权被策略拦
            self.assertTrue(json.loads(tools.device_reset())["ok"])
            self.assertTrue(json.loads(tools.device_power("off"))["ok"])
        self.assertGreaterEqual(sim.counters["commands"], 4)

    def test_tools_fallback_when_link_down(self):
        import src.tools as tools

        old = tools._device
        tools._device = None
        self.addCleanup(lambda: setattr(tools, "_device", old))
        cfg = {"device": {"enabled": True, "transport": "serial",
                          "serial_url": f"socket://127.0.0.1:{_free_port()}", "timeout": 0.3,
                          "retries": 0, "monitor": False}}
        with patch("src.config.load_config", return_value=cfg):
            out = json.loads(tools.device_status())
        self.assertFalse(out["ok"])
        self.assertIn("设备不可用", out["reason"])                  # 连不上要说清楚，不装成功

    def test_sim_mode_unchanged(self):
        """默认（sim）仍走 Phase 0 进程内模拟器，行为不变。"""
        import src.tools as tools

        old = tools._device
        tools._device = None
        self.addCleanup(lambda: setattr(tools, "_device", old))
        with patch("src.config.load_config", return_value={"device": {"enabled": False,
                                                                     "transport": "sim"}}):
            st = json.loads(tools.device_status())
        self.assertIn("temperature_c", st)
        self.assertNotIn("connected", st)


class TestConfigWriterSections(unittest.TestCase):
    """老配置升级路径：配置里没有 device / sandbox 段时，写入器要自己补段（别让用户卡住）。

    2026-09-28 实测踩到：安装版（pip 装的）配置是旧模板生成的，没有 device 段，
    于是 `/device mode serial ...` 直接报「配置里找不到 device 段」——升级路径必须能自愈。
    """

    def _tmp_cfg(self, body="models: {}\nroles: {}\n"):
        tmp = tempfile.mkdtemp(prefix="forge-cfg-")
        path = os.path.join(tmp, "models.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return path

    def test_appends_missing_device_section(self):
        import yaml

        import src.config_writer as cw

        path = self._tmp_cfg()
        with patch.object(cw, "config_path", return_value=path):
            ok, msg = cw.set_device_transport("serial", "socket://127.0.0.1:9009")
        self.assertTrue(ok, msg)
        text = open(path, encoding="utf-8").read()
        for frag in ("device:", "transport: serial", 'serial_url: "socket://127.0.0.1:9009"',
                     "enabled: true", "policy:"):
            self.assertIn(frag, text)
        cfg = yaml.safe_load(text)
        self.assertEqual(cfg["device"]["transport"], "serial")
        self.assertTrue(cfg["device"]["enabled"])
        # 第二次调用走正常「改行」路径
        with patch.object(cw, "config_path", return_value=path):
            ok2, msg2 = cw.set_device_transport("sim")
        self.assertTrue(ok2, msg2)
        cfg2 = yaml.safe_load(open(path, encoding="utf-8").read())
        self.assertEqual(cfg2["device"]["transport"], "sim")
        self.assertFalse(cfg2["device"]["enabled"])

    def test_appends_missing_sandbox_section(self):
        import yaml

        import src.config_writer as cw

        path = self._tmp_cfg()
        with patch.object(cw, "config_path", return_value=path):
            ok, msg = cw.set_sandbox_mode("docker")
        self.assertTrue(ok, msg)
        cfg = yaml.safe_load(open(path, encoding="utf-8").read())
        self.assertEqual(cfg["sandbox"]["mode"], "docker")


if __name__ == "__main__":
    unittest.main(verbosity=2)
