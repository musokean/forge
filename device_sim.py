"""美容仪**设备端**模拟器（硬件 Phase 1：说真协议，不是直调 Python 方法）。

Phase 0 的 `fake_device.py` 是「进程内直调方法」；本文件是**设备端固件的行为替身**：
它跑 `forge/hwproto.py` 的协议（line-JSON + CRC + seq/ack/state），因此 Agent 侧走的是
**真正的传输 + 协议 + 控制平面**，只是最后的物理器件是模拟的。固件骨架见
`hardware/esp32_beauty_device.ino`（同一套协议，换成真继电器/传感器即可）。

支持三种承载（同一套协议，换承载不改逻辑）：
    socket    TCP 透传服务端（默认 :9009）—— 没有硬件也能验真链路：
              Agent 侧 `serial_for_url("socket://127.0.0.1:9009")` 就是串口服务器/esp-link 的同一套代码路径
    serial    真串口（COM5 / /dev/ttyUSB0），直接连真设备或 USB-TTL
    mqtt      局域网多设备（订阅 {prefix}/cmd/{device}，发布 {prefix}/up/{device}）

用法：
    python device_sim.py                                  # 默认 TCP 服务端 :9009
    python device_sim.py --transport socket --port 9009 --test-hooks
    python device_sim.py --transport serial --url COM5 --baud 115200
    python device_sim.py --transport mqtt --host 127.0.0.1 --device-id beauty-01

    # 另一侧（Agent）：
    #   config device: {transport: serial, serial_url: "socket://127.0.0.1:9009"}
    #   或 REPL 里 /device connect socket://127.0.0.1:9009

测试钩子（`--test-hooks`，默认关；仅用于自动化验证，别在真机上开）：
    sim_set_temp {value: 44.9}   直接设定温度（快速制造过热场景）
    sim_trip                      立即触发过热保护
    sim_heat {delta: 5}           温度加 delta
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_device import BeautyDevice                       # noqa: E402  物理模型（Phase 0 复用）
from forge import hwproto as P                                # noqa: E402  真协议

DEFAULT_CAPABILITIES = ["status", "power_on", "power_off", "set_level", "reset_safety"]
TEST_CAPABILITIES = ["sim_set_temp", "sim_trip", "sim_heat"]


class DeviceSimulator:
    """设备端：把命令落到物理模型上，回 ack + state，异常发 event。"""

    def __init__(self, device=None, device_id="beauty-01", fw="sim-1.0",
                 test_hooks=False, verbose=True):
        self.dev = device or BeautyDevice()
        self.device_id = device_id
        self.fw = fw
        self.test_hooks = bool(test_hooks)
        self.verbose = verbose
        self.seq_base = 0
        self._tripped_reported = bool(self.dev.status().get("safety_tripped"))
        self.counters = {"frames_in": 0, "frames_out": 0, "commands": 0, "errors": 0}

    # ---------- 工具 ----------
    def _log(self, msg):
        if self.verbose:
            print(f"  [device] {msg}", flush=True)

    def capabilities(self):
        caps = list(DEFAULT_CAPABILITIES)
        if self.test_hooks:
            caps += TEST_CAPABILITIES
        return caps

    def limits(self):
        return {"max_level": max(self.dev.LEVEL_CURRENT) if hasattr(self.dev, "LEVEL_CURRENT") else 3,
                "max_temp_c": float(self.dev.MAX_SAFE_TEMP),
                "levels": sorted(k for k in self.dev.LEVEL_CURRENT if k > 0)}

    def _safety_dict(self):
        return {"tripped": bool(self.dev.status().get("safety_tripped")),
                "max_temp_c": float(self.dev.MAX_SAFE_TEMP)}

    # ---------- 应用命令（物理层动作）----------
    def _apply(self, cmd_id, args):
        """返回 (ok, reason, code)；**只有这里碰物理模型**。"""
        if cmd_id == "status":
            return True, "", ""
        if cmd_id == "power_on":
            r = self.dev.power_on()
        elif cmd_id == "power_off":
            r = self.dev.power_off()
        elif cmd_id == "set_level":
            try:
                lv = int(args.get("level"))
            except Exception:
                return False, f"level 必须是整数，收到 {args.get('level')!r}", P.E_BAD_JSON
            r = self.dev.set_level(lv)
        elif cmd_id == "reset_safety":
            r = self.dev.reset_safety()
        elif cmd_id == "sim_set_temp" and self.test_hooks:
            with self.dev._lock:                                   # 测试钩子：直达物理模型
                self.dev._temp = float(args.get("value", 25.0))
            return True, "测试钩子：温度已设定", ""
        elif cmd_id == "sim_heat" and self.test_hooks:
            with self.dev._lock:
                self.dev._temp += float(args.get("delta", 1.0))
            return True, "测试钩子：温度已加热", ""
        elif cmd_id == "sim_trip" and self.test_hooks:
            with self.dev._lock:
                self.dev._temp = float(self.dev.MAX_SAFE_TEMP) + 0.1
            return True, "测试钩子：已制造过热条件", ""
        else:
            return False, f"设备不支持命令 {cmd_id!r}", P.E_UNSUPPORTED
        if isinstance(r, dict) and r.get("ok") is False:
            return False, r.get("reason") or "设备拒绝", P.E_REJECTED
        return True, "", ""

    # ---------- 帧处理（协议层）----------
    def handle(self, frame: dict) -> list:
        """处理一帧，返回要回给 Agent 的帧列表（bytes）。"""
        self.counters["frames_in"] += 1
        t = frame.get("type")
        seq = int(frame.get("seq") or 0)
        out = []
        if t == P.MSG_HELLO:
            out.append(P.hello_ack(seq, self.device_id, self.dev.name, fw=self.fw,
                                   capabilities=self.capabilities(), limits=self.limits(),
                                   safety=self._safety_dict()))
            self._log(f"握手：客户端 {frame.get('client')!r} → 回报身份与能力")
        elif t == P.MSG_CMD:
            cmd_id = str(frame.get("id") or "")
            args = dict(frame.get("args") or {})
            self.counters["commands"] += 1
            ok, reason, code = self._apply(cmd_id, args)
            # 先 ack（设备接下了 / 拒绝了），再 state（**真实生效后的状态**）
            out.append(P.ack(seq, cmd_id, ok=ok, reason=reason, code=code))
            if ok:
                out.append(P.state(seq, cmd_id, self.dev.status(), ok=True))
            self._log(f"命令 {cmd_id}{args} → {'OK' if ok else '拒绝: ' + reason}")
        elif t == P.MSG_PING:
            out.append(P.pong(seq))
        else:
            self.counters["errors"] += 1
            self._log(f"忽略不支持的消息类型 {t!r}")
        # 过热事件（只在状态跳变时报一次，模拟固件中断上报）
        tripped = bool(self.dev.status().get("safety_tripped"))
        if tripped and not self._tripped_reported:
            self._tripped_reported = True
            out.append(P.event("over_temp", {"temperature_c": self.dev.status().get("temperature_c"),
                                             "limit": self.dev.MAX_SAFE_TEMP}))
            self._log("⚡ 过热保护触发 → 已上报 event")
        elif not tripped:
            self._tripped_reported = False
        self.counters["frames_out"] += len(out)
        return out

    # ---------- 循环 ----------
    def serve_stream(self, transport, duration=None, idle_timeout=120.0):
        """在任意 Transport 上跑协议循环（socket / 串口 / MQTT 共用）。"""
        reader = P.FrameReader()
        t0 = time.time()
        last_rx = time.time()
        while True:
            if duration and (time.time() - t0) > duration:
                self._log("到时退出")
                return self.counters
            if time.time() - last_rx > idle_timeout:
                self._log(f"空闲 {idle_timeout}s 无数据，退出")
                return self.counters
            chunk = None
            try:
                chunk = transport.read_line(timeout=0.3)
            except EOFError:                 # 对端断开：本次连接正常结束（调用方决定是否继续等接入）
                self._log("对端断开")
                return self.counters
            except Exception as e:
                self._log(f"读失败：{e}")
                return self.counters
            if not chunk:
                continue
            last_rx = time.time()
            try:
                frames = reader.feed(chunk)
            except P.ProtocolError as e:
                self._log(f"坏帧（{e.code}）：{e.message}")
                self.counters["errors"] += 1
                continue
            for f in frames:
                for resp in self.handle(f):
                    transport.write(resp)


# ══════════════════════════════════════════════════════════════════
# 承载：TCP 透传服务端 / 真串口 / MQTT
# ══════════════════════════════════════════════════════════════════
class StreamTransport:
    """把「一行一行读写」包成 Transport 的形状（给 serve_stream 用）。"""

    def __init__(self, reader_fn, writer_fn, closer=None, name="stream"):
        self._read = reader_fn
        self._write = writer_fn
        self._close = closer
        self.name = name

    def read_line(self, timeout=0.3):
        return self._read(timeout)

    def write(self, data):
        return self._write(data)

    def close(self):
        if self._close:
            self._close()


def serve_socket(sim: DeviceSimulator, port=9009, host="127.0.0.1", duration=None, once=True):
    """TCP 透传服务端：pyserial 的 `socket://host:port` 直接连（等价串口服务器/esp-link）。"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(1)
    print(f"  [device] TCP 透传已监听 {host}:{port}（Agent 侧用 socket://{host}:{port}）", flush=True)
    t_start = time.time()
    try:
        # 支持「多次接入」：客户端断开后继续等下一次（断线重连 / 多轮命令都要用）
        while not duration or (time.time() - t_start) < duration:
            try:
                conn, addr = srv.accept()
            except (socket.timeout, KeyboardInterrupt):
                continue
            except OSError:
                break
            print(f"  [device] 客户端接入 {addr}", flush=True)
            conn.settimeout(0.3)
            buf = bytearray()

            def read_line(timeout=0.3, _c=conn, _b=buf):
                try:
                    data = _c.recv(4096)
                    if not data:
                        return b"__EOF__"          # 对端关了：本连接结束
                    _b.extend(data)
                except socket.timeout:
                    return None
                except OSError:
                    return b"__EOF__"
                i = _b.find(b"\n")
                if i < 0:
                    return None
                line = bytes(_b[:i + 1])
                del _b[:i + 1]
                return line

            def _guard_read(timeout=0.3, _rl=read_line):
                line = _rl(timeout)
                if line == b"__EOF__":
                    raise EOFError("client closed")
                return line

            try:
                sim.serve_stream(StreamTransport(_guard_read, conn.sendall, conn.close, f"tcp:{port}"),
                                 duration=duration, idle_timeout=60.0)
            except EOFError:
                pass
            finally:
                if not duration or (time.time() - t_start) < duration:
                    print("  [device] 客户端断开，等待下一次接入…", flush=True)
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        try:
            srv.close()
        except Exception:
            pass
    return dict(sim.counters)


def serve_serial(sim: DeviceSimulator, url, baudrate=115200, duration=None):
    """真串口（COM5 / /dev/ttyUSB0 / loop://）。"""
    from forge.hwtransport import SerialTransport

    tr = SerialTransport(url, baudrate=baudrate, timeout=0.3).open()
    print(f"  [device] 串口已打开 {url} @ {baudrate}", flush=True)
    try:
        return sim.serve_stream(tr, duration=duration)
    finally:
        tr.close()


def serve_mqtt(sim: DeviceSimulator, host="127.0.0.1", port=1883, device_id="beauty-01",
               topic_prefix="forge/dev", qos=1, duration=None):
    """MQTT 承载：订阅 {prefix}/cmd/{device}，发布 {prefix}/up/{device}。"""
    try:
        import paho.mqtt.client as mqtt
    except Exception:
        print("  [device] 未装 paho-mqtt：pip install \"handcraft-agent[device]\"", flush=True)
        return {}
    prefix = topic_prefix.strip("/")
    topic_cmd = f"{prefix}/cmd/{device_id}"
    topic_up = f"{prefix}/up/{device_id}"
    reader = P.FrameReader()
    holder = {}

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"forge-sim-{device_id}")
    except Exception:
        client = mqtt.Client(client_id=f"forge-sim-{device_id}")

    def on_connect(c, u, flags, rc, *a):
        c.subscribe(topic_cmd, qos=qos)
        print(f"  [device] MQTT 已连 {host}:{port}，订阅 {topic_cmd}，上报 {topic_up}", flush=True)

    def on_message(c, u, msg):
        for f in reader.feed(msg.payload + b"\n"):
            for resp in sim.handle(f):
                c.publish(topic_up, resp, qos=qos)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, int(port), 30)
    client.loop_start()
    holder["client"] = client
    try:
        t0 = time.time()
        while not duration or (time.time() - t0) < duration:
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass
    return sim.counters


def main():
    ap = argparse.ArgumentParser(description="美容仪设备端模拟器（硬件 Phase 1，说真协议）")
    ap.add_argument("--transport", default="socket", choices=["socket", "serial", "mqtt"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9009)
    ap.add_argument("--url", default="COM3", help="serial 承载用：COM5 / /dev/ttyUSB0 / loop://")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--device-id", default="beauty-01")
    ap.add_argument("--topic-prefix", default="forge/dev")
    ap.add_argument("--test-hooks", action="store_true", help="开启 sim_* 测试钩子（仅自动化验证）")
    ap.add_argument("--duration", type=float, default=None, help="运行多少秒后退出（脚本化测试用）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sim = DeviceSimulator(device_id=args.device_id, test_hooks=args.test_hooks, verbose=not args.quiet)
    print(f"  [device] 「{sim.dev.name}」模拟设备端启动 · 承载 {args.transport} · 协议 v{P.PROTOCOL_VERSION}", flush=True)
    if args.transport == "socket":
        counters = serve_socket(sim, port=args.port, host=args.host, duration=args.duration)
    elif args.transport == "serial":
        counters = serve_serial(sim, args.url, baudrate=args.baud, duration=args.duration)
    else:
        counters = serve_mqtt(sim, host=args.host, port=args.port, device_id=args.device_id,
                              topic_prefix=args.topic_prefix, duration=args.duration)
    print(f"  [device] 退出 · 统计 {json.dumps(counters, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
