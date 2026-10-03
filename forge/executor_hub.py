"""#17 客户端执行器 · 中心侧枢纽（A25「服务器大脑 + 分布式手脚」的大脑侧接线）。

形态（对齐 A25）：**客户端主动连上来（反向连接）**，所以目标 PC 不用开任何入站端口：

    客户端执行器（每台 PC 一台）  ──HTTP 长轮询（出站）──▶  中心 hub（本模块）
        register  → 声明身份与能力
        poll      → 领命令（挂起等待，最长 `poll_wait` 秒）
        result    → 回结果（成功/失败 + 输出 + 可选截图/文件内容）

能力（能力清单由**客户端自己声明**，中心再按策略二次过滤）：

    shell       执行命令（客户端侧走 `forge/sandbox.py`，不是裸跑）
    read_file   读文件（路径 jail + 体积上限）
    write_file  写文件
    list_dir    列目录
    screenshot  截屏（返回 base64 PNG + 可选 OCR 文本；需客户端装 GUI 依赖）
    input       GUI 操作：click / move / type / key / scroll（需 pyautogui）

安全（A25「分级安全」，与 A06 一脉相承）：
  · 中心侧：`executor.hub.allow_caps` 白名单 + 每设备声明能力取交集 + 命令超时 + 体积上限 + 全量审计
  · 客户端侧：`executor.client.*` 能力白名单 / 路径 jail / 体积上限 / shell 走沙箱
  · **分阶段放权**：`stage = readonly | low_risk | approval | closed_loop`
      readonly  只允许 screenshot / read_file / list_dir
      low_risk  再加 shell / write_file（低风险写）
      approval  写类动作要带 `approved: true`（人工确认后才能过）
      closed_loop 全放（前提是鉴权/审计/超时/回滚都成熟）
"""
from __future__ import annotations

import base64
import os
import queue
import threading
import time
import uuid

CAPS = ("shell", "read_file", "write_file", "list_dir", "screenshot", "input")
READONLY_CAPS = ("read_file", "list_dir", "screenshot")
WRITE_CAPS = ("shell", "write_file", "input")
STAGES = ("readonly", "low_risk", "approval", "closed_loop")

DEFAULT_HUB = {
    "enabled": True,
    "stage": "low_risk",
    "allow_caps": list(CAPS),
    "poll_wait": 25.0,          # 长轮询挂起秒数（客户端一次占住这么久，空转就返回）
    "command_timeout": 30.0,    # 中心等结果的上限
    "max_result_bytes": 512 * 1024,
    "stale_seconds": 120,       # 多久没心跳算离线
    "max_queue": 32,
}


def _log(level, event, **fields):
    try:
        from .logging_setup import log_event

        log_event(level, event, **fields)
    except Exception:
        pass


class ExecutorError(Exception):
    def __init__(self, code, message=""):
        self.code = code
        self.message = message or code
        super().__init__(f"{code}: {self.message}")


class ExecutorInfo:
    """一台客户端执行器的身份与状态。"""

    def __init__(self, device_id, host="", os_name="", version="", capabilities=None,
                 stage=None, max_bytes=None, tags=None):
        self.device_id = str(device_id)
        self.host = host
        self.os = os_name
        self.version = version
        self.capabilities = [c for c in (capabilities or []) if c in CAPS]
        self.stage = stage or ""
        self.max_bytes = int(max_bytes or 0)
        self.tags = dict(tags or {})
        self.registered_at = time.time()
        self.last_seen = time.time()
        self.commands_run = 0
        self.failures = 0
        self.busy = False

    def touch(self):
        self.last_seen = time.time()

    def as_dict(self):
        return {"device_id": self.device_id, "host": self.host, "os": self.os,
                "version": self.version, "capabilities": self.capabilities,
                "stage": self.stage, "max_bytes": self.max_bytes, "tags": self.tags,
                "online": self.online(), "last_seen_ago": round(time.time() - self.last_seen, 1),
                "commands_run": self.commands_run, "failures": self.failures}

    def online(self, stale_seconds=None):
        return (time.time() - self.last_seen) <= float(stale_seconds or DEFAULT_HUB["stale_seconds"])


class Command:
    """一条派给客户端的命令（含状态机与结果）。"""

    def __init__(self, device_id, cap, action, args=None, timeout=None, approved=False):
        self.seq = uuid.uuid4().hex[:12]
        self.device_id = device_id
        self.cap = cap
        self.action = action
        self.args = dict(args or {})
        self.approved = bool(approved)
        self.timeout = float(timeout or DEFAULT_HUB["command_timeout"])
        self.created_at = time.time()
        self.sent_at = None
        self.done_at = None
        self.status = "pending"          # pending → sent → done | timeout | error
        self.ok = False
        self.output = ""
        self.error = ""
        self.extra = {}
        self.event = threading.Event()   # dispatch 侧等结果用

    def as_payload(self):
        return {"seq": self.seq, "cap": self.cap, "action": self.action, "args": self.args,
                "approved": self.approved, "timeout": self.timeout,
                "age": round(time.time() - self.created_at, 3)}

    def as_dict(self, with_output=True):
        d = {"seq": self.seq, "device_id": self.device_id, "cap": self.cap, "action": self.action,
             "status": self.status, "ok": self.ok, "error": self.error,
             "ms": round(((self.done_at or time.time()) - self.created_at) * 1000, 1)}
        if with_output:
            d["output"] = self.output
        return d

    def __repr__(self):
        return f"<Command {self.cap}:{self.action} {self.status} ok={self.ok}>"


class ExecutorHub:
    """中心侧枢纽：注册表 + 每设备命令队列 + 结果关联。"""

    def __init__(self, cfg=None, hub_conf=None):
        conf = dict(DEFAULT_HUB)
        if isinstance(cfg, dict):
            conf.update({k: v for k, v in ((cfg.get("executor") or {}).get("hub") or {}).items()
                         if v is not None})
        conf.update({k: v for k, v in (hub_conf or {}).items() if v is not None})
        self.conf = conf
        self.stage = str(conf.get("stage") or "low_risk").lower()
        self.allow_caps = [c for c in (conf.get("allow_caps") or CAPS) if c in CAPS]
        self._lock = threading.Lock()
        self._devices = {}                      # device_id → ExecutorInfo
        self._queues = {}                       # device_id → queue.Queue[Command]
        self._waiting = {}                      # seq → Command（等结果的）
        self.audit_log = []

    # ---------- 能力 / 策略 ----------
    def allowed(self, device: ExecutorInfo, cap, args=None, approved=False):
        """返回 (allowed, reason, code)。中心侧策略在客户端策略之前再拦一道。"""
        args = args or {}
        if cap not in CAPS:
            return False, f"未知能力 {cap!r}（支持 {', '.join(CAPS)}）", "E_UNKNOWN_CAP"
        if cap not in self.allow_caps:
            return False, f"能力 {cap} 被中心策略禁用（executor.hub.allow_caps）", "E_CAP_DENIED"
        if device.capabilities and cap not in device.capabilities:
            return False, f"设备「{device.device_id}」未声明能力 {cap}（它支持：{', '.join(device.capabilities) or '无'}）", "E_CAP_UNSUPPORTED"
        if self.stage == "readonly" and cap not in READONLY_CAPS:
            return False, f"阶段 readonly：只允许 {', '.join(READONLY_CAPS)}", "E_STAGE_READONLY"
        if self.stage == "approval" and cap in WRITE_CAPS and not approved:
            return False, f"阶段 approval：写类动作需要人工确认（approved=true），{cap} 被拦", "E_NEED_APPROVAL"
        max_bytes = int(self.conf.get("max_result_bytes") or 0)
        limit = min(x for x in (max_bytes, device.max_bytes or max_bytes) if x > 0)
        if cap in ("read_file", "write_file") and int(args.get("max_bytes") or 0) > limit:
            return False, f"体积超上限（{args.get('max_bytes')} > {limit}）", "E_TOO_LARGE"
        return True, "", ""

    # ---------- 注册 / 心跳 ----------
    def register(self, payload):
        device_id = str((payload or {}).get("device_id") or "").strip()
        if not device_id:
            raise ExecutorError("E_NO_DEVICE", "注册必须带 device_id")
        with self._lock:
            info = self._devices.get(device_id)
            if info is None:
                info = ExecutorInfo(device_id, host=payload.get("host", ""),
                                    os_name=payload.get("os", ""), version=payload.get("version", ""),
                                    capabilities=payload.get("capabilities"), stage=payload.get("stage"),
                                    max_bytes=payload.get("max_bytes"), tags=payload.get("tags"))
                self._devices[device_id] = info
                self._queues[device_id] = queue.Queue(maxsize=int(self.conf.get("max_queue") or 32))
                _log("INFO", "executor_registered", device=device_id, host=info.host, os=info.os,
                     capabilities=info.capabilities, stage=info.stage)
            else:
                info.touch()
                if payload.get("capabilities"):
                    info.capabilities = [c for c in payload["capabilities"] if c in CAPS]
                for k in ("host", "os", "version", "stage", "max_bytes"):
                    if payload.get(k):
                        setattr(info, k if k != "os" else "os", payload[k])
            info.touch()
            return {"ok": True, "device_id": device_id, "sync": round(time.time(), 3),
                    "poll_wait": float(self.conf.get("poll_wait") or 25.0),
                    "hub_stage": self.stage,
                    "hub_caps": self.allow_caps,
                    "device": info.as_dict()}

    def heartbeat(self, device_id):
        info = self._devices.get(device_id)
        if info:
            info.touch()
        return bool(info)

    def devices(self, only_online=False):
        with self._lock:
            items = list(self._devices.values())
        out = [d.as_dict() for d in items]
        if only_online:
            out = [d for d in out if d["online"]]
        return out

    def prune(self):
        """清掉超时离线的设备（保留统计）。"""
        stale = float(self.conf.get("stale_seconds") or 120)
        with self._lock:
            gone = [k for k, v in self._devices.items() if not v.online(stale)]
            for k in gone:
                self._devices.pop(k, None)
                self._queues.pop(k, None)
        if gone:
            _log("WARNING", "executor_pruned", devices=gone)
        return gone

    # ---------- 客户端：领命令 / 回结果 ----------
    def poll(self, device_id, wait=None, max_commands=1):
        """客户端长轮询：有命令就立刻返回，没有就挂起到 wait 秒。"""
        info = self._devices.get(device_id)
        if info is None:
            raise ExecutorError("E_UNKNOWN_DEVICE", f"设备「{device_id}」未注册（先 register）")
        info.touch()
        q = self._queues.get(device_id)
        if q is None:
            raise ExecutorError("E_UNKNOWN_DEVICE", f"设备「{device_id}」队列不存在")
        wait = float(self.conf.get("poll_wait") if wait is None else wait)
        out = []
        deadline = time.time() + max(0.0, wait)
        while len(out) < int(max_commands or 1):
            timeout = max(0.0, deadline - time.time())
            try:
                cmd = q.get(timeout=timeout if (out or timeout > 0) else None)
            except queue.Empty:
                break
            cmd.sent_at = time.time()
            cmd.status = "sent"
            out.append(cmd.as_payload())
            if time.time() >= deadline:
                break
        return out

    def submit_result(self, device_id, seq, ok, output="", error="", extra=None):
        with self._lock:
            cmd = self._waiting.get(seq)
        if cmd is None:
            _log("WARNING", "executor_result_unknown", device=device_id, seq=seq)
            return False
        cmd.status = "done" if ok else "error"
        cmd.ok = bool(ok)
        cmd.output = str(output or "")
        cmd.error = str(error or "")
        cmd.extra = dict(extra or {})
        cmd.done_at = time.time()
        info = self._devices.get(device_id)
        if info:
            info.touch()
            info.commands_run += 1
            if not ok:
                info.failures += 1
        cmd.event.set()
        self._audit(cmd)
        return True

    # ---------- 中心：派命令 ----------
    def dispatch(self, device_id, cap, action, args=None, timeout=None, approved=False):
        """派一条命令并等结果（同步，供工具/四角色循环调用）。"""
        info = self._devices.get(device_id)
        if info is None:
            raise ExecutorError("E_UNKNOWN_DEVICE",
                                f"设备「{device_id}」不在线（已知：{', '.join(self._devices) or '无'}）。"
                                "客户端执行器要先起来并注册")
        if not info.online(float(self.conf.get("stale_seconds") or 120)):
            raise ExecutorError("E_OFFLINE", f"设备「{device_id}」已离线（最后一次心跳 {round(time.time() - info.last_seen)}s 前）")
        allowed, reason, code = self.allowed(info, cap, args, approved=approved)
        if not allowed:
            cmd = Command(device_id, cap, action, args, timeout, approved)
            cmd.status, cmd.error, cmd.done_at = "error", reason, time.time()
            self._audit(cmd, blocked=True, code=code)
            raise ExecutorError(code, reason)
        cmd = Command(device_id, cap, action, args, timeout, approved)
        with self._lock:
            self._waiting[cmd.seq] = cmd
            q = self._queues.get(device_id)
        if q is None:
            raise ExecutorError("E_UNKNOWN_DEVICE", f"设备「{device_id}」队列不存在")
        try:
            q.put_nowait(cmd)
        except queue.Full:
            with self._lock:
                self._waiting.pop(cmd.seq, None)
            raise ExecutorError("E_QUEUE_FULL", f"设备「{device_id}」待发队列满（{self.conf.get('max_queue')}）")
        got = cmd.event.wait(cmd.timeout)
        with self._lock:
            self._waiting.pop(cmd.seq, None)
        if not got:
            cmd.status = "timeout"
            cmd.error = f"命令超时（{cmd.timeout}s 内没有结果；客户端可能离线/卡住）"
            cmd.done_at = time.time()
            self._audit(cmd, timeout=True, code="E_TIMEOUT")
        return cmd

    # ---------- 审计 ----------
    def _audit(self, cmd, blocked=False, timeout=False, code=""):
        entry = {"t": round(time.time(), 3), "device": cmd.device_id, "cap": cmd.cap,
                 "action": cmd.action, "args": cmd.args, "status": cmd.status, "ok": cmd.ok,
                 "ms": round(((cmd.done_at or time.time()) - cmd.created_at) * 1000, 1),
                 "error": cmd.error, "code": code, "blocked": blocked}
        self.audit_log.append(entry)
        self.audit_log[:] = self.audit_log[-500:]
        level = "WARNING" if (blocked or timeout or not cmd.ok) else "INFO"
        _log(level, "executor_command", **{k: v for k, v in entry.items() if k != "t"})
        return entry

    def audit(self, n=10):
        return self.audit_log[-int(n):]

    def stats(self):
        return {"stage": self.stage, "allow_caps": self.allow_caps,
                "devices": len(self._devices), "online": sum(1 for d in self._devices.values() if d.online()),
                "pending": sum(q.qsize() for q in self._queues.values()),
                "waiting": len(self._waiting), "audit": len(self.audit_log),
                "poll_wait": self.conf.get("poll_wait"), "command_timeout": self.conf.get("command_timeout")}


# ---------- 全局单例（server 与工具共用同一份注册表）----------
_HUB = None
_LOCK = threading.Lock()


def get_hub(cfg=None, refresh=False) -> ExecutorHub:
    global _HUB
    with _LOCK:
        if _HUB is None or refresh:
            if cfg is None:
                try:
                    from .config import load_config

                    cfg = load_config()
                except Exception:
                    cfg = {}
            _HUB = ExecutorHub(cfg)
        return _HUB


def reset_hub():
    global _HUB
    with _LOCK:
        _HUB = None


def encode_bytes(data, limit=None) -> str:
    """把字节/base64 编码（带体积上限），供 read_file / screenshot 结果回传。"""
    if isinstance(data, str):
        raw = data.encode("utf-8")
        b64 = data
    else:
        raw = data or b""
        b64 = base64.b64encode(raw).decode("ascii")
    if limit and len(raw) > int(limit):
        raise ExecutorError("E_TOO_LARGE", f"内容 {len(raw)} 字节超过上限 {limit}")
    return b64
