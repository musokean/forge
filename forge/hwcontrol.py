"""#16 硬件 Phase 1 · 控制平面（A24 点名的生产级真正难点）。

直接让 Agent 发消息 = 演示，不是生产。缺了控制平面会出三件事：设备别名漂移 / 跨站点误控、
ACK 语义不一（「发出去了」≠「生效了」）。本模块就是那层控制平面，五层对位：

    MCP 工具层    tools.py 的 device_* 工具（提出请求）
    资产目录      AssetRegistry —— 别名 → 设备身份/传输/站点，**未知别名直接拒绝，不猜**
    策略引擎      PolicyEngine —— 分阶段放权（只读/低风险/审批/闭环）+ 限额 + 冷却 + 越权即拒
    命令服务      CommandService —— 状态机 Created→Sent→Accepted→Applied，含超时/重试/回滚/审计
    MQTT·串口网关  hwtransport —— 协议转换 + 断线重连

对外只有 `HardwareLink` 一个门面，**接口与 Phase 0 的 `fake_device.BeautyDevice` 一致**
（status / power_on / power_off / set_level / reset_safety + name），
所以 `tools.py` 一行都不用改就能从模拟器切到真链路（鸭子类型）。

安全默认值（对齐 A24「分阶段放权，别一上来就闭环」）：
    · 默认 `stage: low_risk`，写命令仍走 forge 审批层（本模块不越权替你自动放行）
    · `stage: readonly` 时任何写命令都在**发出去之前**被拒
    · 温度越线 / 运行超时 → 控制平面主动断电（不只依赖设备自身的保护）
    · 非回环地址（真·远程 MQTT）默认拒绝，要显式 `policy.allow_remote: true`

配置（`config/models.yaml` 的 `device` 段）：

    device:
      enabled: false
      transport: sim            # sim | serial | mqtt
      device_id: beauty-01
      serial_url: socket://127.0.0.1:9009    # 或 COM5 / loop://
      baudrate: 115200
      timeout: 3
      retries: 2
      monitor: true             # 后台巡检（限额执行 / 心跳）
      monitor_interval: 5
      policy:
        stage: low_risk         # readonly | low_risk | approval | closed_loop
        max_level: 3
        max_temp_c: 45
        max_runtime_s: 900
        cooldown_s: 1
        allow_remote: false
"""
from __future__ import annotations

import threading
import time

from . import hwproto as P
from .hwproto import ProtocolError
from .hwtransport import SerialTransport, TransportError, open_transport

# 命令状态机
ST_CREATED = "Created"
ST_SENT = "Sent"
ST_ACCEPTED = "Accepted"
ST_APPLIED = "Applied"
ST_REJECTED = "Rejected"
ST_TIMEOUT = "Timeout"

WRITE_COMMANDS = ("power_on", "power_off", "set_level", "reset_safety")

DEFAULT_POLICY = {
    "stage": "low_risk",
    "max_level": 3,
    "max_temp_c": 45.0,
    "max_runtime_s": 900,
    "cooldown_s": 1.0,
    "allow_remote": False,
}


def _log(level, event, **fields):
    try:
        from .logging_setup import log_event

        log_event(level, event, **fields)
    except Exception:
        pass


class HardwareError(Exception):
    """控制平面错误（未连接 / 未知设备 / 越权 / 协议失败）。"""

    def __init__(self, code, message=""):
        self.code = code
        self.message = message or code
        super().__init__(f"{code}: {self.message}")


# ══════════════════════════════════════════════════════════════════
# ① 资产目录
# ══════════════════════════════════════════════════════════════════
class AssetRegistry:
    """别名 → 设备资产。**未知别名一律拒绝**（防别名漂移/跨站点误控）。"""

    def __init__(self, assets=None, default_id="beauty-01", transport="sim"):
        self.assets = {}
        for alias, item in (assets or {}).items():
            if isinstance(item, dict):
                self.assets[str(alias)] = {"device_id": item.get("device_id", str(alias)),
                                           "transport": item.get("transport", transport),
                                           "site": item.get("site", ""),
                                           "capabilities": list(item.get("capabilities") or [])}
        if default_id and not self.assets:
            self.assets[str(default_id)] = {"device_id": str(default_id), "transport": transport,
                                            "site": "", "capabilities": []}

    def resolve(self, alias=None) -> dict:
        key = str(alias or "").strip()
        if not key:
            if len(self.assets) == 1:
                return next(iter(self.assets.values()))
            raise HardwareError("E_UNKNOWN_DEVICE",
                                f"未指定设备别名，且资产目录里有 {len(self.assets)} 台设备")
        if key in self.assets:
            return self.assets[key]
        for a in self.assets.values():                      # 按 device_id 反查（别名与 id 都认）
            if a["device_id"] == key:
                return a
        raise HardwareError("E_UNKNOWN_DEVICE",
                            f"资产目录里没有「{key}」（已知：{', '.join(sorted(self.assets))}）")

    def all(self) -> list:
        return [{"alias": k, **v} for k, v in sorted(self.assets.items())]


# ══════════════════════════════════════════════════════════════════
# ② 策略引擎
# ══════════════════════════════════════════════════════════════════
class PolicyEngine:
    """分阶段放权 + 限额 + 冷却。返回 (allowed, reason, code)。"""

    def __init__(self, policy=None):
        conf = dict(DEFAULT_POLICY)
        conf.update({k: v for k, v in (policy or {}).items() if v is not None})
        self.conf = conf
        self.stage = str(conf.get("stage") or "low_risk").lower()
        self._last_write_at = 0.0

    def stage_allows_write(self) -> bool:
        return self.stage != "readonly"

    def check(self, cmd_id, args=None, state=None, remote=False, force=False):
        args = args or {}
        state = state or {}
        if remote and not self.conf.get("allow_remote"):
            return False, "策略拒绝：远程（非本机）链路默认不可写（要开请设 policy.allow_remote: true）", "E_REMOTE_BLOCKED"
        if cmd_id not in WRITE_COMMANDS or force:
            return True, "", ""
        if not self.stage_allows_write():
            return False, f"策略拒绝：当前阶段 {self.stage} 只读，写命令一律拦截", "E_STAGE_READONLY"
        if state.get("safety_tripped") and cmd_id != "reset_safety":
            return False, "设备安全保护已触发，只能先 reset_safety 复位", "E_SAFETY_TRIPPED"
        temp = float(state.get("temperature_c") or 0)
        if temp >= float(self.conf.get("max_temp_c") or 999) and cmd_id != "power_off":
            return False, f"温度 {temp}°C 已达上限 {self.conf['max_temp_c']}°C，拒绝写入（只允许关机）", "E_OVER_TEMP"
        if cmd_id == "set_level":
            lv = args.get("level")
            if not isinstance(lv, int) or not 1 <= lv <= int(self.conf.get("max_level") or 3):
                return False, f"档位 {lv!r} 越权（允许 1..{self.conf.get('max_level')}）", "E_LEVEL_LIMIT"
        cool = float(self.conf.get("cooldown_s") or 0)
        if cool > 0 and (time.time() - self._last_write_at) < cool:
            return False, f"冷却中（{cool}s 内不允许连续写），稍后再试", "E_COOLDOWN"
        return True, "", ""

    def note_write(self):
        self._last_write_at = time.time()

    def status(self) -> dict:
        return dict(self.conf)


# ══════════════════════════════════════════════════════════════════
# ③ 命令服务（状态机 + 超时 + 重试 + 回滚 + 审计）
# ══════════════════════════════════════════════════════════════════
class CommandResult:
    def __init__(self, cmd_id, state, ok=False, reason="", code="", attempts=0, ms=0.0,
                 state_after=None, rolled_back=False):
        self.id = cmd_id
        self.state = state
        self.ok = ok
        self.reason = reason
        self.code = code
        self.attempts = attempts
        self.ms = ms
        self.state_after = state_after or {}
        self.rolled_back = rolled_back

    def as_dict(self):
        return {"ok": self.ok, "id": self.id, "state": self.state, "reason": self.reason,
                "code": self.code, "attempts": self.attempts, "ms": round(self.ms, 1),
                "rolled_back": self.rolled_back, "state": self.state,
                **({"state_after": self.state_after} if self.state_after else {})}

    def __repr__(self):
        return f"<CommandResult {self.id} {self.state} ok={self.ok} {self.ms:.0f}ms>"


class CommandService:
    """命令状态机：Created → Sent → Accepted → Applied（或 Rejected / Timeout）。

    「收到 ack」只说明设备**接下**了命令；**只有 state 快照才算生效**——这正是 A24 说的
    「ACK 语义不一 / 不知道是否生效」的解法。
    """

    def __init__(self, transport, timeout=3.0, retries=2):
        self.transport = transport
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._seq = 0
        self._lock = threading.Lock()
        self.events = []            # 设备侧异步事件（over_temp 等）

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def send(self, frame: bytes) -> int:
        return self.transport.write(frame)

    # ---- 读一帧（带超时），交给调用方判定 ----
    def _read(self, deadline):
        reader = getattr(self, "_reader", None)
        if reader is None:
            reader = self._reader = P.FrameReader()
        while time.time() < deadline:
            chunk = self.transport.read_line(timeout=min(0.5, max(0.05, deadline - time.time())))
            if not chunk:
                continue
            for frame in reader.feed(chunk):
                if frame.get("type") == P.MSG_EVENT:
                    self.events.append(frame)
                    _log("WARNING", "device_event", kind=frame.get("kind"),
                         detail=frame.get("detail"), seq=frame.get("seq"))
                return frame
        return None

    def wait_for(self, seq, timeout=None):
        """等到与本命令 seq 相关的帧（ack / state）。返回 (kind, frame) 或 (None, None)。"""
        deadline = time.time() + (timeout if timeout is not None else self.timeout)
        while True:
            frame = self._read(deadline)
            if frame is None:
                return None, None
            if frame.get("type") in (P.MSG_ACK, P.MSG_STATE) and int(frame.get("seq") or 0) == int(seq):
                return frame["type"], frame

    def execute(self, cmd_id, args=None, timeout=None, retries=None) -> CommandResult:
        args = args or {}
        timeout = float(timeout or self.timeout)
        retries = self.retries if retries is None else int(retries)
        t0 = time.time()
        seq = self.next_seq()
        attempts = 0
        res = CommandResult(cmd_id, ST_CREATED)
        while attempts <= retries:
            attempts += 1
            try:
                self.send(P.cmd(cmd_id, args, seq=seq))
            except TransportError as e:                  # 链路断了：控平面如实报错，不装成功
                res.state, res.reason, res.code = "Failed", str(e), "E_TRANSPORT"
                break
            res.state = ST_SENT
            deadline = time.time() + timeout
            applied = False
            while time.time() < deadline:
                kind, frame = self.wait_for(seq, timeout=max(0.1, deadline - time.time()))
                if kind is None:
                    break
                if kind == P.MSG_ACK:
                    if frame.get("ok"):
                        res.state = ST_ACCEPTED
                        if frame.get("state"):
                            res.state_after = frame["state"]
                    else:
                        res.state = ST_REJECTED
                        res.reason = frame.get("reason") or "设备拒绝"
                        res.code = frame.get("code") or P.E_REJECTED
                        applied = True                          # 明确失败，不用再等
                        break
                elif kind == P.MSG_STATE:                        # 只有 state 才算 Applied
                    res.state = ST_APPLIED
                    res.ok = bool(frame.get("ok", True))
                    res.state_after = frame.get("state") or {}
                    res.reason = frame.get("reason") or ""
                    applied = True
                    break
            if applied:
                break
            if res.state == ST_ACCEPTED:
                # 设备接了但没给 state —— 再等一轮，仍无则按「未确认生效」处理
                res.reason = res.reason or "设备已接收（ack）但未回报状态快照"
                res.code = res.code or "E_NO_STATE"
        if res.state == ST_SENT or (res.state == ST_ACCEPTED and not res.ok):
            res.state = ST_TIMEOUT if res.state == ST_SENT else res.state
            res.code = res.code or P.E_TIMEOUT
            res.reason = res.reason or f"命令超时（{timeout}s × {attempts} 次）"
            res.ok = False
        res.ms = (time.time() - t0) * 1000
        res.attempts = attempts
        _log("INFO" if res.ok else "WARNING", "device_cmd",
             cmd=cmd_id, args=args, state=res.state, ok=res.ok, attempts=attempts,
             ms=res.ms, reason=res.reason, code=res.code)
        return res


# ══════════════════════════════════════════════════════════════════
# ④ 门面：与 fake_device.BeautyDevice 同接口
# ══════════════════════════════════════════════════════════════════
class HardwareLink:
    """真链路门面。接口与 `fake_device.BeautyDevice` 一致（鸭子类型），tools.py 无需改动。"""

    def __init__(self, cfg=None, transport=None, alias=None):
        conf = ((cfg or {}).get("device") or {}) if isinstance(cfg, dict) else {}
        self.conf = conf
        self.registry = AssetRegistry(conf.get("assets"), default_id=conf.get("device_id", "beauty-01"),
                                      transport=conf.get("transport", "sim"))
        self.asset = self.registry.resolve(alias or conf.get("device_id"))
        self.policy = PolicyEngine(conf.get("policy"))
        self.timeout = float(conf.get("timeout") or 3.0)
        self.retries = int(conf.get("retries") or 2)
        self.monitor_enabled = bool(conf.get("monitor", True))
        self.monitor_interval = float(conf.get("monitor_interval") or 5.0)
        self._transport = transport or open_transport(cfg if isinstance(cfg, dict) else None,
                                                      timeout=self.timeout)
        # 与 BeautyDevice 同名的公开属性
        self.name = self.asset.get("device_id") or "beauty-01"
        self.device_info = {}
        self.capabilities = list(self.asset.get("capabilities") or [])
        self.limits = {}
        self._svc = CommandService(self._transport, timeout=self.timeout, retries=self.retries)
        self._state = {}
        self._connected = False
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
        self.audit_log = []          # 本进程内的命令审计（结构化日志另有全量副本）

    # ---------- 连接 / 握手 ----------
    def connect(self):
        try:
            self._transport.open()
        except TransportError as e:
            raise HardwareError("E_TRANSPORT",
                                f"{e} —— 检查 URL/波特率/接线；没有真硬件时先起模拟设备端："
                                "python device_sim.py（默认 127.0.0.1:9009）")
        seq = self._svc.next_seq()
        self._svc.send(P.hello(client="forge", capabilities=list(WRITE_COMMANDS), seq=seq))
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            frame = self._svc._read(deadline)
            if frame and frame.get("type") == P.MSG_HELLO_ACK:
                self.device_info = {k: frame.get(k) for k in ("device_id", "name", "fw")}
                self.capabilities = list(frame.get("capabilities") or self.capabilities)
                self.limits = dict(frame.get("limits") or {})
                self.name = frame.get("name") or self.name
                self._connected = True
                _log("INFO", "device_connected", device=self.name,
                     transport=self._transport.describe(), fw=frame.get("fw"),
                     capabilities=self.capabilities, limits=self.limits)
                if self.monitor_enabled:
                    self._start_monitor()
                return self.status()
        self._connected = False
        raise HardwareError("E_NO_HANDSHAKE",
                            f"设备未回应握手（{self.timeout}s，链路 {self._transport.describe()}）——"
                            "检查 URL/波特率/设备是否上电，或用 device_sim.py 起个模拟设备端")

    def close(self):
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        self._monitor_thread = None
        try:
            self._transport.close()
        except Exception:
            pass
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self._transport.connected

    def reconnect(self):
        self.close()
        self._stop_monitor.clear()
        return self.connect()

    # ---------- 只读 ----------
    def status(self) -> dict:
        """读状态（只读，零风险）：以设备回报的真实状态为准。"""
        if not self._connected:
            return {"device": self.name, "connected": False, "power": False, "level": 0,
                    "temperature_c": 0.0, "current_a": 0.0, "run_seconds": 0.0,
                    "safety_tripped": False, "transport": self._transport.describe()}
        res = self._svc.execute("status", {}, timeout=self.timeout, retries=max(0, self.retries - 1))
        if res.state_after:
            self._state = dict(res.state_after)
        self._apply_device_safety()
        self._enforce_limits()
        return self._snapshot()

    def _snapshot(self) -> dict:
        """本地状态快照（不发命令）——`_write` 回包里用它，避免 status()→_enforce_limits()→_write 递归。"""
        return {"device": self.name, "connected": self.connected,
                "transport": self._transport.describe(),
                **{k: self._state.get(k) for k in ("power", "level", "temperature_c", "current_a",
                                                   "run_seconds", "safety_tripped")}}

    # ---------- 写（策略 → 状态机 → 回滚）----------
    def _write(self, cmd_id, args=None, rollback=None, force=False) -> dict:
        args = args or {}
        if not self._connected:
            return {"ok": False, "reason": "设备未连接（先 /device connect）", "code": "E_NOT_CONNECTED"}
        remote = self._is_remote()
        allowed, reason, code = self.policy.check(cmd_id, args, self._state, remote=remote, force=force)
        if not allowed:
            _log("WARNING", "device_policy_block", cmd=cmd_id, args=args, code=code, reason=reason)
            self._audit(cmd_id, args, "Blocked", False, reason, code)
            return {"ok": False, "reason": reason, "code": code, "blocked": True}
        if cmd_id in WRITE_COMMANDS:
            self.policy.note_write()
        res = self._svc.execute(cmd_id, args)
        self._audit(cmd_id, args, res.state, res.ok, res.reason, res.code)
        if res.state_after:
            self._state = dict(res.state_after)
        if not res.ok and rollback and res.state in (ST_TIMEOUT, ST_REJECTED):
            rb_id, rb_args = rollback
            rb = self._svc.execute(rb_id, rb_args)
            res.rolled_back = rb.ok
            _log("WARNING", "device_rollback", cmd=cmd_id, rollback=rb_id, args=rb_args, ok=rb.ok)
        out = {"ok": res.ok, "state": res.state, "reason": res.reason or "",
               "cmd": cmd_id, "attempts": res.attempts, "ms": round(res.ms, 1)}
        if res.rolled_back:
            out["rolled_back"] = True
        if res.ok:
            out["status"] = self._snapshot()      # 快照，不再发起一次往返（也不会递归）
        return out

    def power_on(self) -> dict:
        return self._write("power_on", {}, rollback=("power_off", {}))

    def power_off(self) -> dict:
        return self._write("power_off", {})

    def set_level(self, level: int) -> dict:
        prev = int(self._state.get("level") or 0)
        rb = ("set_level", {"level": prev}) if prev else None
        return self._write("set_level", {"level": level}, rollback=rb)

    def reset_safety(self) -> dict:
        return self._write("reset_safety", {})

    # ---------- 限额执行 / 安全 ----------
    def _apply_device_safety(self):
        if self._state.get("safety_tripped"):
            _log("ERROR", "device_safety_tripped", device=self.name,
                 temperature_c=self._state.get("temperature_c"))

    def _enforce_limits(self) -> list:
        """控制平面自己执行限额（不只依赖设备端保护）：超温 / 超时 → 主动断电。"""
        acted = []
        if not self._connected or not self._state:
            return acted
        temp = float(self._state.get("temperature_c") or 0)
        run_s = float(self._state.get("run_seconds") or 0)
        max_temp = float(self.policy.conf.get("max_temp_c") or 999)
        max_run = float(self.policy.conf.get("max_runtime_s") or 0)
        power = bool(self._state.get("power"))
        if power and max_temp and temp >= max_temp:
            _log("ERROR", "device_guard_over_temp", temperature_c=temp, limit=max_temp)
            self._write("power_off", {}, force=True)
            acted.append("over_temp_power_off")
        elif power and max_run and run_s > max_run:
            _log("WARNING", "device_guard_max_runtime", run_seconds=run_s, limit=max_run)
            self._write("power_off", {}, force=True)
            acted.append("max_runtime_power_off")
        return acted

    def _is_remote(self) -> bool:
        """链路是否「远程」（非本机回环）——远程默认不可写。"""
        t = self._transport
        if not isinstance(t, SerialTransport):
            return False
        url = (getattr(t, "url", "") or "").lower()
        if url.startswith("socket://") or url.startswith("rfc2217://"):
            host = url.split("://", 1)[1].split(":")[0]
            return host not in ("127.0.0.1", "localhost", "::1", "")
        return False

    def _audit(self, cmd_id, args, state, ok, reason="", code=""):
        entry = {"t": round(time.time(), 3), "device": self.name, "cmd": cmd_id, "args": args,
                 "state": state, "ok": bool(ok), "reason": reason, "code": code,
                 "by": self.conf.get("actor", "agent")}
        self.audit_log.append(entry)
        self.audit_log[:] = self.audit_log[-200:]
        return entry

    # ---------- 后台巡检 ----------
    def _start_monitor(self):
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        def loop():
            while not self._stop_monitor.wait(self.monitor_interval):
                try:
                    st = self._svc.execute("status", {})
                    if st.state_after:
                        self._state = dict(st.state_after)
                        self._enforce_limits()
                except Exception as e:                   # 巡检线程绝不因异常死掉拖垮进程
                    _log("WARNING", "device_monitor_error", error=str(e))

        self._monitor_thread = threading.Thread(target=loop, name="forge-device-monitor", daemon=True)
        self._monitor_thread.start()

    # ---------- 给 /device 命令看的状态 ----------
    def status_report(self) -> dict:
        return {
            "name": self.name,
            "connected": self.connected,
            "transport": self._transport.describe(),
            "remote": self._is_remote(),
            "device_info": self.device_info,
            "capabilities": self.capabilities,
            "device_limits": self.limits,
            "policy": self.policy.status(),
            "state": dict(self._state),
            "audit_tail": self.audit_log[-5:],
            "monitor": self.monitor_enabled and self._monitor_thread is not None,
        }

    def audit(self, n=10) -> list:
        return self.audit_log[-int(n):]


# ── 全局单例 ──────────────────────────────────────────────────────
_LINK = None
_LOCK = threading.Lock()


def get_link(cfg=None, refresh=False) -> HardwareLink:
    """取全局真链路（首次按配置初始化；refresh 重读配置）。"""
    global _LINK
    with _LOCK:
        if _LINK is None or refresh:
            if cfg is None:
                try:
                    from .config import load_config

                    cfg = load_config()
                except Exception:
                    cfg = {}
            if _LINK is not None:
                try:
                    _LINK.close()
                except Exception:
                    pass
            _LINK = HardwareLink(cfg)
        return _LINK


def reset_link():
    global _LINK
    with _LOCK:
        if _LINK is not None:
            try:
                _LINK.close()
            except Exception:
                pass
        _LINK = None
