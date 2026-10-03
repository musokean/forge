"""#16 硬件 Phase 1 · 协议层（对齐 A24「控制平面五层」的 MQTT/网关与命令服务之间的那一层）。

一条消息 = 一行 JSON（UTF-8，`\\n` 结尾）。串口与 MQTT 共用同一套帧——**换传输不改协议**，
设备端固件（`hardware/esp32_beauty_device.ino`）、模拟器（`device_sim.py`）与 Agent 侧
（`forge/hwcontrol.py`）说的是同一种话。

    {"v":1,"seq":7,"ts":1759000000.12,"type":"cmd","id":"set_level","args":{"level":2},"crc":"3f9a1c2b0d4e"}

字段
    v     协议版本（当前 1；不匹配直接拒绝，见 E_VERSION）
    seq   序号（agent 侧自增，设备回复原样带回 → 用来把回复配到命令上）
    ts    时间戳（float 秒，可选）
    type  消息类型（见下）
    crc   crc32(不含 crc 字段的规范化 JSON) 取前 12 位 hex —— 串口线上会掉字节，
         没有校验就会出现「明明发的是 2 档，设备跳成 3 档」这种最危险的静默错控

消息类型
    hello       agent → device   握手（协议版本 + 客户端标识 + 关心的能力）
    hello_ack   device → agent   设备身份与能力：device_id / name / fw / capabilities / limits
    cmd         agent → device   命令（id + args）
    ack         device → agent   命令**已接收/被拒**（ok/reason）——「收到了」不等于「生效了」
    state       device → agent   状态快照（应用后的真实状态）——**只有 state 才算 applied**
    event       device → agent   异步事件：over_temp / fault / heartbeat
    ping/pong   保活

命令状态机（`CommandService`，A24 点名的「ACK 语义不一」就是靠这两段区分开的）

    Created ──发送──> Sent ──收到 ack.ok──> Accepted ──收到 state──> Applied
                          └── ack 拒绝 ──> Rejected
                          └── 超时/重试用尽 ──> Timeout

错误码
    E_BAD_JSON / E_CRC / E_VERSION / E_UNKNOWN_TYPE / E_UNSUPPORTED / E_REJECTED / E_BUSY / E_TIMEOUT
"""
from __future__ import annotations

import json
import time
import zlib

PROTOCOL_VERSION = 1
MAX_LINE = 64 * 1024          # 单帧上限（超过视为线路异常，断开重连，别把内存吃光）
_CRC_LEN = 12

MSG_HELLO = "hello"
MSG_HELLO_ACK = "hello_ack"
MSG_CMD = "cmd"
MSG_ACK = "ack"
MSG_STATE = "state"
MSG_EVENT = "event"
MSG_PING = "ping"
MSG_PONG = "pong"

TYPES = (MSG_HELLO, MSG_HELLO_ACK, MSG_CMD, MSG_ACK, MSG_STATE, MSG_EVENT, MSG_PING, MSG_PONG)

E_BAD_JSON = "E_BAD_JSON"
E_CRC = "E_CRC"
E_VERSION = "E_VERSION"
E_UNKNOWN_TYPE = "E_UNKNOWN_TYPE"
E_UNSUPPORTED = "E_UNSUPPORTED"
E_REJECTED = "E_REJECTED"
E_BUSY = "E_BUSY"
E_TIMEOUT = "E_TIMEOUT"


class ProtocolError(Exception):
    """协议层错误（坏帧 / 校验失败 / 版本不符 / 未知类型）。"""

    def __init__(self, code, message=""):
        self.code = code
        self.message = message or code
        super().__init__(f"{code}: {self.message}")


def crc_of(body: dict) -> str:
    """帧校验：对**不含 crc 字段**的规范化 JSON 取 crc32，取前 12 位 hex。"""
    payload = {k: v for k, v in body.items() if k != "crc"}
    canon = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return format(zlib.crc32(canon.encode("utf-8")) & 0xFFFFFFFF, "08x")[:_CRC_LEN]


def encode(type_, seq=0, args=None, ts=None, **extra) -> bytes:
    """打包成一帧（bytes，含结尾 `\\n`）。"""
    body = {"v": PROTOCOL_VERSION, "seq": int(seq), "type": type_}
    if ts is not None:
        body["ts"] = round(float(ts), 3)
    if args:
        body["args"] = args
    if extra:
        body.update(extra)
    body["crc"] = crc_of(body)
    return (json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode(raw, verify_crc=True) -> dict:
    """解析一帧（bytes/str）→ dict。坏帧抛 ProtocolError。"""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ProtocolError(E_BAD_JSON, "帧不是合法 UTF-8")
    raw = raw.strip()
    if not raw:
        raise ProtocolError(E_BAD_JSON, "空帧")
    try:
        body = json.loads(raw)
    except Exception:
        raise ProtocolError(E_BAD_JSON, f"JSON 解析失败：{raw[:80]!r}")
    if not isinstance(body, dict):
        raise ProtocolError(E_BAD_JSON, "帧必须是 JSON 对象")
    if verify_crc:
        got = str(body.get("crc") or "")
        if not got:
            raise ProtocolError(E_BAD_JSON, "帧缺少 crc 字段（不是本协议的帧）")
        want = crc_of(body)
        if got != want:
            raise ProtocolError(E_CRC, f"校验不符（收到 {got}，应为 {want}）")
    v = body.get("v")
    if v != PROTOCOL_VERSION:
        raise ProtocolError(E_VERSION, f"协议版本 {v!r} 不被支持（当前支持 {PROTOCOL_VERSION}）")
    t = body.get("type")
    if t not in TYPES:
        raise ProtocolError(E_UNKNOWN_TYPE, f"未知消息类型 {t!r}")
    return body


class FrameReader:
    """把字节流切成完整帧（串口读到的是一段一段的，不能假设一次读到一整帧）。

    - 处理半帧、粘包（一次读到多帧）、垃圾行
    - 单帧超过 `max_line` → 抛 ProtocolError（线路异常时快速失败而不是撑爆内存）
    """

    def __init__(self, max_line=MAX_LINE, verify_crc=True):
        self.buf = bytearray()
        self.max_line = max_line
        self.verify_crc = verify_crc
        self.dropped = 0        # 丢弃的坏帧计数（可观测）

    def feed(self, chunk) -> list:
        """喂入字节 → 返回本次能解析出的所有帧（dict 列表）。"""
        if chunk:
            self.buf.extend(chunk)
        out = []
        while True:
            i = self.buf.find(b"\n")
            if i < 0:
                if len(self.buf) > self.max_line:
                    self.buf.clear()
                    raise ProtocolError(E_BAD_JSON, f"单帧超过 {self.max_line} 字节（无换行）")
                break
            line = bytes(self.buf[:i])
            del self.buf[:i + 1]
            if not line.strip():
                continue
            try:
                out.append(decode(line, verify_crc=self.verify_crc))
            except ProtocolError:
                self.dropped += 1
                continue
        return out


# ── 消息构造便捷函数（agent 侧与设备侧共用，避免两边各写一份字段名）──────
def hello(client="forge", capabilities=None, seq=0) -> bytes:
    return encode(MSG_HELLO, seq=seq, ts=time.time(),
                  client=client, want=list(capabilities or []))


def hello_ack(seq, device_id, name, fw="", capabilities=None, limits=None, safety=None) -> bytes:
    return encode(MSG_HELLO_ACK, seq=seq, ts=time.time(), ok=True, device_id=device_id,
                  name=name, fw=fw, capabilities=list(capabilities or []),
                  limits=dict(limits or {}), safety=dict(safety or {}))


def cmd(cmd_id, args=None, seq=0) -> bytes:
    return encode(MSG_CMD, seq=seq, ts=time.time(), id=cmd_id, args=dict(args or {}))


def ack(seq, cmd_id, ok=True, reason="", code="" , state=None) -> bytes:
    return encode(MSG_ACK, seq=seq, ts=time.time(), id=cmd_id, ok=bool(ok),
                  reason=reason, code=code, state=state or {})


def state(seq, cmd_id, state_dict, ok=True, reason="") -> bytes:
    return encode(MSG_STATE, seq=seq, ts=time.time(), id=cmd_id, ok=bool(ok),
                  reason=reason, state=dict(state_dict or {}))


def event(kind, detail=None, seq=0) -> bytes:
    return encode(MSG_EVENT, seq=seq, ts=time.time(), kind=kind, detail=dict(detail or {}))


def ping(seq=0) -> bytes:
    return encode(MSG_PING, seq=seq, ts=time.time())


def pong(seq=0) -> bytes:
    return encode(MSG_PONG, seq=seq, ts=time.time())
