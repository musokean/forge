"""#16 硬件 Phase 1 · 传输层：同一套协议，换传输不改上层（A24「硬件抽象层：串口 / MQTT」）。

三种传输：
  · `SerialTransport`  —— 串口/UART。**用 pyserial 的 URL 机制**，所以一个类覆盖四种真实场景：
        COM5              Windows 真串口
        /dev/ttyUSB0      Linux 真串口
        socket://host:port  串口服务器 / ser2net / ESP32 的 TCP 透传（**没有硬件也能验真链路**）
        loop://           回环，自测传输层本身
        rfc2217://host:port  RFC2217 远程串口
  · `MqttTransport`    —— 局域网多设备（WiFi/IoT）。下行 `{prefix}/cmd/{device}`，
        上行 `{prefix}/up/{device}`（ack / state / event 都走这条），QoS 可配。
  · `MemoryTransport`  —— 进程内成对队列（模拟器直连 / 单元测试），零依赖。

共性：`open() / close() / write(bytes) / read_line(timeout) / connected / describe()`。
**没有换行缓冲的重连**由上层 `HardwareLink` 负责（`reconnect()`）。

依赖：pyserial 仅串口需要、paho-mqtt 仅 MQTT 需要 —— 都做可选依赖
（`pip install "handcraft-agent[device]"`），不装也不影响 forge 主流程。
"""
from __future__ import annotations

import queue
import threading
import time

try:                                       # 可选依赖：只有用串口才需要
    import serial                          # type: ignore
    from serial import serial_for_url      # type: ignore
except Exception:                          # pragma: no cover
    serial = None
    serial_for_url = None

try:                                       # 可选依赖：只有用 MQTT 才需要
    import paho.mqtt.client as mqtt        # type: ignore
except Exception:                          # pragma: no cover
    mqtt = None


class TransportError(Exception):
    """传输层错误（打不开 / 断开 / 写失败）。"""


class Transport:
    """传输抽象：写字节、按行读、描述自己。"""

    scheme = "base"

    def __init__(self, timeout=2.0):
        self.timeout = float(timeout)
        self._opened = False
        self.last_error = ""

    # ---- 生命周期 ----
    def open(self):                                        # pragma: no cover - 子类实现
        raise NotImplementedError

    def close(self):                                       # pragma: no cover - 子类实现
        self._opened = False

    # ---- 读写 ----
    def write(self, data: bytes) -> int:                   # pragma: no cover - 子类实现
        raise NotImplementedError

    def read_line(self, timeout=None):                     # pragma: no cover - 子类实现
        """返回一行 bytes（含 `\n`）或 None（超时）。"""
        raise NotImplementedError

    @property
    def connected(self) -> bool:
        return self._opened

    def describe(self) -> str:                             # pragma: no cover - 子类实现
        return self.scheme


# ══════════════════════════════════════════════════════════════════
# 串口（真串口 / socket 桥 / loop 自测，同一套 pyserial 代码路径）
# ══════════════════════════════════════════════════════════════════
class SerialTransport(Transport):
    """pyserial URL 传输：COM5 / /dev/ttyUSB0 / socket://… / loop:// / rfc2217://。"""

    scheme = "serial"

    def __init__(self, url, baudrate=115200, timeout=2.0, write_timeout=None):
        super().__init__(timeout=timeout)
        self.url = str(url or "")
        self.baudrate = int(baudrate or 115200)
        self.write_timeout = write_timeout
        self.ser = None

    @property
    def needs_pyserial(self) -> bool:
        return True

    def open(self):
        if serial_for_url is None:
            raise TransportError("未装 pyserial —— 串口链路需要：pip install \"handcraft-agent[device]\"")
        if not self.url:
            raise TransportError("串口 URL 为空（例：COM5 / socket://127.0.0.1:9009 / loop://）")
        try:
            self.ser = serial_for_url(self.url, baudrate=self.baudrate, timeout=self.timeout,
                                      write_timeout=self.write_timeout or self.timeout)
        except Exception as e:
            self.last_error = str(e)
            raise TransportError(f"打开 {self.url} 失败：{e}")
        self._opened = True
        return self

    def close(self):
        ser, self.ser = self.ser, None
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass
        # Windows 上 socket:// 的句柄在 GC 时会被 pyserial 再关一次 → 抛 WinError 6 噪音，
        # 这里把内部 socket 置空，避免「Exception ignored in」污染 CLI/CI 输出
        try:
            if getattr(ser, "sock", None) is not None:
                ser.sock = None
        except Exception:
            pass
        self._opened = False

    def write(self, data: bytes) -> int:
        if self.ser is None:
            raise TransportError("串口未打开")
        try:
            n = self.ser.write(data)
            return int(n or 0)
        except Exception as e:
            self.last_error = str(e)
            self._opened = False
            raise TransportError(f"串口写入失败：{e}")

    def read_line(self, timeout=None):
        if self.ser is None:
            return None
        if timeout is not None:
            self.ser.timeout = float(timeout)
        try:
            return self.ser.readline()
        except Exception as e:
            self.last_error = str(e)
            self._opened = False
            raise TransportError(f"串口读取失败：{e}")

    def describe(self):
        return f"serial[{self.url} @ {self.baudrate}]"


# ══════════════════════════════════════════════════════════════════
# MQTT（局域网多设备）
# ══════════════════════════════════════════════════════════════════
class MqttTransport(Transport):
    """MQTT 传输：下行 `{prefix}/cmd/{device}`，上行 `{prefix}/up/{device}`。

    `client` 参数可注入（测试用假 broker 客户端；生产用 paho）。
    """

    scheme = "mqtt"

    def __init__(self, host="127.0.0.1", port=1883, device_id="beauty-01",
                 topic_prefix="forge/dev", qos=1, client_id="", username="", password="",
                 timeout=5.0, keepalive=30, client=None):
        super().__init__(timeout=timeout)
        self.host = host
        self.port = int(port or 1883)
        self.device_id = device_id
        self.prefix = str(topic_prefix or "forge/dev").strip("/")
        self.qos = int(qos)
        self.client_id = client_id or f"forge-agent-{int(time.time()) % 100000}"
        self.username = username
        self.password = password
        self.keepalive = int(keepalive)
        self.topic_down = f"{self.prefix}/cmd/{self.device_id}"
        self.topic_up = f"{self.prefix}/up/{self.device_id}"
        self._inbox = queue.Queue()
        self._client = client
        self._own_client = client is None

    def _build_client(self):
        if mqtt is None:
            raise TransportError("未装 paho-mqtt —— MQTT 链路需要：pip install \"handcraft-agent[device]\"")
        try:                                        # paho 2.x 需要显式回调版本
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        except Exception:                           # paho 1.x
            c = mqtt.Client(client_id=self.client_id)
        if self.username:
            c.username_pw_set(self.username, self.password)
        c.on_connect = self._on_connect
        c.on_message = self._on_message
        return c

    # ---- paho 回调（1.x/2.x 参数不同，用 *args 兼容）----
    def _on_connect(self, *args):
        try:
            self._client.subscribe(self.topic_up, qos=self.qos)
            self._opened = True
        except Exception as e:                       # pragma: no cover
            self.last_error = str(e)

    def _on_message(self, client, userdata, message):
        payload = getattr(message, "payload", None)
        if payload:
            self._inbox.put(payload if payload.endswith(b"\n") else payload + b"\n")

    # ---- 生命周期 ----
    def open(self):
        if self._client is None:
            self._client = self._build_client()
        else:
            # 注入的 client（测试假 broker / 复用的客户端）也要接回调，否则收不到上行帧
            self._client.on_message = self._on_message
            self._client.on_connect = self._on_connect
        try:
            if self._own_client:
                self._client.connect(self.host, self.port, self.keepalive)
                self._client.loop_start()
            else:
                self._client.connect(self.host, self.port, self.keepalive)
            # 注入的假客户端不会触发 on_connect → 主动订阅，保证两条路径一致
            if not self._opened:
                self._client.subscribe(self.topic_up, qos=self.qos)
            self._opened = True
        except TransportError:
            raise
        except Exception as e:
            self.last_error = str(e)
            raise TransportError(f"MQTT 连接 {self.host}:{self.port} 失败：{e}")
        return self

    def close(self):
        try:
            if self._client is not None:
                if self._own_client:
                    self._client.loop_stop()
                self._client.disconnect()
        except Exception:
            pass
        self._opened = False

    def write(self, data: bytes) -> int:
        if self._client is None:
            raise TransportError("MQTT 未连接")
        try:
            self._client.publish(self.topic_down, data, qos=self.qos)
            return len(data)
        except Exception as e:
            self.last_error = str(e)
            self._opened = False
            raise TransportError(f"MQTT 发布失败：{e}")

    def read_line(self, timeout=None):
        try:
            return self._inbox.get(timeout=timeout if timeout is not None else self.timeout)
        except queue.Empty:
            return None

    def describe(self):
        return f"mqtt[{self.host}:{self.port} {self.topic_down} → {self.topic_up} qos={self.qos}]"


# ══════════════════════════════════════════════════════════════════
# 进程内成对队列（模拟器直连 / 测试）
# ══════════════════════════════════════════════════════════════════
class MemoryTransport(Transport):
    """成对的内存管道：`MemoryTransport.pair()` 返回 (a, b)，a 写的 b 能读到。"""

    scheme = "memory"

    def __init__(self, name="mem", timeout=2.0, write_queue=None, read_queue=None):
        super().__init__(timeout=timeout)
        self.name = name
        self._out = write_queue if write_queue is not None else queue.Queue()
        self._in = read_queue if read_queue is not None else queue.Queue()

    @classmethod
    def pair(cls, timeout=2.0):
        q_ab, q_ba = queue.Queue(), queue.Queue()
        return (cls("a", timeout, q_ab, q_ba), cls("b", timeout, q_ba, q_ab))

    def open(self):
        self._opened = True
        return self

    def close(self):
        self._opened = False

    def write(self, data: bytes) -> int:
        self._out.put(data)
        return len(data)

    def read_line(self, timeout=None):
        try:
            return self._in.get(timeout=timeout if timeout is not None else self.timeout)
        except queue.Empty:
            return None

    def describe(self):
        return f"memory[{self.name}]"


# ══════════════════════════════════════════════════════════════════
# 工厂
# ══════════════════════════════════════════════════════════════════
def open_transport(target=None, timeout=2.0, **kw) -> Transport:
    """按 URL/配置开传输：

        mqtt://host:1883            → MqttTransport
        sim / memory://             → MemoryTransport
        COM5 / loop:// / socket://… → SerialTransport（pyserial URL）
        None 且 cfg 里有 device 段   → 按 device.transport 自动选
    """
    cfg = target if isinstance(target, dict) else None
    if cfg is not None:
        conf = cfg.get("device") or {}
        mode = str(conf.get("transport") or "sim").lower()
        if mode == "mqtt":
            mq = conf.get("mqtt") or {}
            return MqttTransport(host=mq.get("host", "127.0.0.1"), port=mq.get("port", 1883),
                                 device_id=conf.get("device_id", "beauty-01"),
                                 topic_prefix=mq.get("topic_prefix", "forge/dev"),
                                 qos=mq.get("qos", 1), username=mq.get("username", ""),
                                 password=mq.get("password", ""),
                                 timeout=conf.get("timeout", timeout))
        if mode in ("sim", "memory", "fake"):
            return MemoryTransport("sim", timeout=timeout)
        target = conf.get("serial_url") or conf.get("url") or ""
        if not target:
            raise TransportError("device.transport 为串口但没配 device.serial_url")
        return SerialTransport(target, baudrate=conf.get("baudrate", 115200),
                               timeout=conf.get("timeout", timeout))

    t = str(target or "").strip()
    low = t.lower()
    if not t or low in ("sim", "memory", "memory://", "fake"):
        return MemoryTransport("sim", timeout=timeout)
    if low.startswith("mqtt://") or low.startswith("mqtts://") or low.startswith("tcp://"):
        rest = low.split("://", 1)[1]
        host, _, port = rest.partition(":")
        return MqttTransport(host=host or "127.0.0.1", port=int(port or 1883), timeout=timeout, **kw)
    return SerialTransport(t, baudrate=kw.pop("baudrate", 115200), timeout=timeout, **kw)
