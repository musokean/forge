"""#17 客户端执行器 · 客户端侧（装在被控 PC 上，只出站，不开放入站端口）。

对齐 A25：**复用现成组件，别自己造远程桌面**——
  · 远程画面/远控底座：RustDesk（本模块不重造，见 `docs/executor.md` 的对接说明）
  · 屏幕理解：截图回传中心，由中心的 Computer Use 视觉角色判断（`src/cua.py`）
  · GUI 操作：pyautogui（可选依赖，未装则该能力**如实报不可用**，不假装成功）

能力（客户端**自己声明**，中心再按策略过滤）：

    shell      执行命令 —— **走本机 `src/sandbox.py`**（Docker 可用则容器隔离，否则加固本机执行）
    read_file / write_file / list_dir   —— 路径 jail（默认限定 `executor.client.root`）+ 体积上限
    screenshot 截屏 → base64 PNG（+ 可选 OCR 文本）
    input      click / move / type / key / scroll

客户端侧策略（`executor.client`）：能力白名单 · 路径 jail · 体积上限 · shell 走沙箱。
中心侧还有一道（`executor.hub`）：白名单 + 分阶段放权 + 超时 + 审计。**两道都要有**。

跑法：

    python executor_agent.py --center http://127.0.0.1:8080 --token <APIKEY> --id pc-01
"""
from __future__ import annotations

import base64
import json
import os
import platform
import time

from .executor_hub import CAPS, READONLY_CAPS, STAGES, WRITE_CAPS, ExecutorError

DEFAULT_CLIENT = {
    "allow_caps": list(CAPS),
    "root": "",                  # 路径 jail；默认 = 执行器启动目录
    "stage": "low_risk",
    "max_bytes": 256 * 1024,
    "poll_wait": 25.0,
    "deny_patterns": [],
    "version": "exec-1.0",
}


# ══════════════════════════════════════════════════════════════════
# 屏幕/输入驱动：真实现需可选依赖；未装 → 能力如实报不可用
# ══════════════════════════════════════════════════════════════════
class ScreenDriver:
    """GUI 驱动抽象（Win/macOS 一致接口，底层按 OS 分派——A25「跨平台抽象层」）。"""

    kind = "base"

    def probe(self):
        return False, "未实现"

    def screenshot(self) -> bytes:                       # pragma: no cover - 子类实现
        raise ExecutorError("E_NO_GUI", "截图驱动不可用")

    def click(self, x, y, button="left", clicks=1):       # pragma: no cover
        raise ExecutorError("E_NO_GUI", "输入驱动不可用")

    def move(self, x, y):                                 # pragma: no cover
        raise ExecutorError("E_NO_GUI", "输入驱动不可用")

    def type_text(self, text, interval=0.02):             # pragma: no cover
        raise ExecutorError("E_NO_GUI", "输入驱动不可用")

    def key(self, keys):                                  # pragma: no cover
        raise ExecutorError("E_NO_GUI", "输入驱动不可用")

    def scroll(self, amount):                             # pragma: no cover
        raise ExecutorError("E_NO_GUI", "输入驱动不可用")

    def size(self):
        return (0, 0)


class NullDriver(ScreenDriver):
    """没有 GUI 依赖时的占位：所有动作**明确失败**（绝不假装成功）。"""

    kind = "unavailable"

    def __init__(self, reason="未安装 GUI 依赖（pip install \"handcraft-agent[executor]\"）"):
        self.reason = reason

    def probe(self):
        return False, self.reason

    def screenshot(self):
        raise ExecutorError("E_NO_GUI", self.reason)

    def click(self, x, y, button="left", clicks=1):
        raise ExecutorError("E_NO_GUI", self.reason)

    def move(self, x, y):
        raise ExecutorError("E_NO_GUI", self.reason)

    def type_text(self, text, interval=0.02):
        raise ExecutorError("E_NO_GUI", self.reason)

    def key(self, keys):
        raise ExecutorError("E_NO_GUI", self.reason)

    def scroll(self, amount):
        raise ExecutorError("E_NO_GUI", self.reason)


class PyAutoGuiDriver(ScreenDriver):
    """真 GUI 驱动（pyautogui + Pillow）。未装依赖 → 自动退化为 NullDriver。"""

    kind = "pyautogui"

    def __init__(self):
        self._gui = None
        self._img = None
        try:
            import pyautogui            # type: ignore
            import io                   # noqa: F401
            from PIL import Image       # type: ignore  # noqa: F401
            self._gui = pyautogui
            pyautogui.FAILSAFE = True   # 鼠标撞左上角即中止（防失控）
            pyautogui.PAUSE = 0.05
        except Exception as e:
            self._err = f"GUI 依赖不可用（{type(e).__name__}）：pip install \"handcraft-agent[executor]\""

    def probe(self):
        return (self._gui is not None), (getattr(self, "_err", "") or "ok")

    def screenshot(self) -> bytes:
        if self._gui is None:
            raise ExecutorError("E_NO_GUI", getattr(self, "_err", "GUI 不可用"))
        import io

        shot = self._gui.screenshot()
        buf = io.BytesIO()
        shot.save(buf, format="PNG")
        return buf.getvalue()

    def click(self, x, y, button="left", clicks=1):
        if self._gui is None:
            raise ExecutorError("E_NO_GUI", getattr(self, "_err", "GUI 不可用"))
        self._gui.click(int(x), int(y), clicks=int(clicks or 1), button=button or "left")

    def move(self, x, y):
        if self._gui is None:
            raise ExecutorError("E_NO_GUI", getattr(self, "_err", "GUI 不可用"))
        self._gui.moveTo(int(x), int(y))

    def type_text(self, text, interval=0.02):
        if self._gui is None:
            raise ExecutorError("E_NO_GUI", getattr(self, "_err", "GUI 不可用"))
        self._gui.typewrite(str(text), interval=float(interval or 0))

    def key(self, keys):
        if self._gui is None:
            raise ExecutorError("E_NO_GUI", getattr(self, "_err", "GUI 不可用"))
        parts = [p.strip() for p in str(keys).split("+") if p.strip()]
        if len(parts) > 1:
            self._gui.hotkey(*parts)
        elif parts:
            self._gui.press(parts[0])

    def scroll(self, amount):
        if self._gui is None:
            raise ExecutorError("E_NO_GUI", getattr(self, "_err", "GUI 不可用"))
        self._gui.scroll(int(amount or 0))

    def size(self):
        if self._gui is None:
            return (0, 0)
        return tuple(self._gui.size())


class FakeDriver(ScreenDriver):
    """测试/自检用：记录动作、返回固定 PNG，不需要任何 GUI 依赖。"""

    kind = "fake"

    def __init__(self, png=b"\x89PNG\r\n\x1a\nFAKE", size=(1920, 1080)):
        self.png = png
        self._size = size
        self.actions = []

    def probe(self):
        return True, "fake driver"

    def screenshot(self):
        return self.png

    def click(self, x, y, button="left", clicks=1):
        self.actions.append(("click", int(x), int(y), button, int(clicks)))

    def move(self, x, y):
        self.actions.append(("move", int(x), int(y)))

    def type_text(self, text, interval=0.02):
        self.actions.append(("type", str(text)))

    def key(self, keys):
        self.actions.append(("key", str(keys)))

    def scroll(self, amount):
        self.actions.append(("scroll", int(amount)))

    def size(self):
        return self._size


def default_driver():
    """按可用性挑驱动：装了 GUI 依赖用真驱动，否则 NullDriver（能力如实报不可用）。"""
    d = PyAutoGuiDriver()
    ok, _ = d.probe()
    return d if ok else NullDriver(getattr(d, "_err", "GUI 不可用"))


# ══════════════════════════════════════════════════════════════════
# 客户端策略
# ══════════════════════════════════════════════════════════════════
class ClientPolicy:
    """客户端侧策略：能力白名单 + 路径 jail + 体积上限 + 分阶段放权。"""

    def __init__(self, conf=None, root=None):
        c = dict(DEFAULT_CLIENT)
        c.update({k: v for k, v in (conf or {}).items() if v is not None})
        self.conf = c
        self.stage = str(c.get("stage") or "low_risk").lower()
        self.allow_caps = [x for x in (c.get("allow_caps") or CAPS) if x in CAPS]
        self.root = os.path.abspath(root or c.get("root") or os.getcwd())
        self.max_bytes = int(c.get("max_bytes") or DEFAULT_CLIENT["max_bytes"])

    def check_cap(self, cap, approved=False):
        """检查能力是否允许。**注意 approved 必须传进来**——漏了它 approval 阶段会把
        已人工确认的动作也一并拦掉（2026-09-28 自测踩到）。"""
        if cap not in CAPS:
            return False, f"未知能力 {cap!r}", "E_UNKNOWN_CAP"
        if cap not in self.allow_caps:
            return False, f"能力 {cap} 被本机策略禁用（executor.client.allow_caps）", "E_CAP_DENIED"
        if self.stage == "readonly" and cap not in READONLY_CAPS:
            return False, f"本机阶段 readonly：只允许 {', '.join(READONLY_CAPS)}", "E_STAGE_READONLY"
        if self.stage == "approval" and cap in WRITE_CAPS and not approved:
            return False, f"本机阶段 approval：写类动作需人工确认（approved=true），{cap} 被拦", "E_NEED_APPROVAL"
        return True, "", ""

    def safe_path(self, path):
        """路径 jail：解析后必须落在 root 之内（防 ../ 与符号链接逃逸）。"""
        if not path:
            raise ExecutorError("E_BAD_ARGS", "缺少 path")
        p = os.path.realpath(os.path.join(self.root, str(path)) if not os.path.isabs(str(path))
                             else str(path))
        root = os.path.realpath(self.root)
        if p != root and not p.startswith(root + os.sep):
            raise ExecutorError("E_PATH_DENIED", f"路径越界：{p} 不在允许目录 {root} 内")
        return p

    def check_size(self, n, what="内容"):
        if n > self.max_bytes:
            raise ExecutorError("E_TOO_LARGE", f"{what} {n} 字节超过本机上限 {self.max_bytes}")


# ══════════════════════════════════════════════════════════════════
# 客户端执行器
# ══════════════════════════════════════════════════════════════════
class ExecutorClient:
    """客户端：注册 → 长轮询领命令 → 本地执行 → 回结果。只出站。"""

    def __init__(self, center="http://127.0.0.1:8080", token="", device_id="",
                 client_conf=None, driver=None, http=None, sandbox=None, verbose=False):
        self.center = str(center or "").rstrip("/")
        self.token = str(token or "")
        self.device_id = str(device_id or f"{platform.node() or 'pc'}-{os.getpid()}")
        self.policy = ClientPolicy(client_conf)
        self.driver = driver if driver is not None else default_driver()
        self.verbose = verbose
        self._sandbox = sandbox
        self._http = http
        self.registered = False
        self.stop_flag = False
        self.stats = {"polled": 0, "executed": 0, "failed": 0, "blocked": 0, "errors": 0}
        self.local_audit = []

    # ---------- HTTP ----------
    def _client(self):
        if self._http is None:
            try:
                import httpx
            except Exception as e:                       # pragma: no cover
                raise ExecutorError("E_NO_HTTPX", f"需要 httpx 才能连中心：{e}")
            headers = {"X-API-Key": self.token} if self.token else {}
            self._http = httpx.Client(base_url=self.center, headers=headers, timeout=30.0)
        return self._http

    def _post(self, path, payload):
        r = self._client().post(path, json=payload)
        if r.status_code >= 400:
            detail = ""
            try:
                detail = r.json().get("detail") or r.json().get("error") or ""
            except Exception:
                detail = r.text[:200]
            raise ExecutorError(f"E_HTTP_{r.status_code}", f"{path} → {r.status_code} {detail}")
        try:
            return r.json()
        except Exception:
            return {}

    # ---------- 能力 ----------
    def capabilities(self):
        """声明能力：GUI 依赖没装就不声明 screenshot/input（**不假装有**）。"""
        caps = [c for c in self.policy.allow_caps if c in ("shell", "read_file", "write_file", "list_dir")]
        ok, _ = self.driver.probe()
        if ok:
            caps += [c for c in ("screenshot", "input") if c in self.policy.allow_caps]
        return caps

    def register(self):
        payload = {"device_id": self.device_id, "host": platform.node(), "os": platform.platform(),
                   "version": self.policy.conf.get("version", "exec-1.0"),
                   "capabilities": self.capabilities(), "stage": self.policy.stage,
                   "max_bytes": self.policy.max_bytes,
                   "tags": {"python": platform.python_version(), "driver": self.driver.kind,
                            "root": self.policy.root}}
        out = self._post("/api/executor/register", payload)
        self.registered = bool(out.get("ok"))
        if self.verbose:
            print(f"  [exec] 已注册 {self.device_id} · 能力 {payload['capabilities']} · "
                  f"driver={self.driver.kind} · root={self.policy.root}", flush=True)
        return out

    def _sandbox_run(self, command):
        if self._sandbox is None:
            from .sandbox import get_sandbox

            self._sandbox = get_sandbox()
        res = self._sandbox.run(command)
        return res

    # ---------- 执行 ----------
    def handle(self, cap, action, args=None, approved=False):
        """本地执行一条命令 → (ok, output, error, extra)。**所有异常都转成失败结果。**"""
        args = dict(args or {})
        ok, reason, code = self.policy.check_cap(cap, approved=approved)
        if not ok:
            self.stats["blocked"] += 1
            return False, "", f"{code}: {reason}", {"blocked": True, "code": code}
        try:
            return self._handle(cap, action, args)
        except ExecutorError as e:
            self.stats["failed"] += 1
            return False, "", f"{e.code}: {e.message}", {"code": e.code}
        except Exception as e:                                # 兜底：绝不把异常抛回去
            self.stats["errors"] += 1
            return False, "", f"E_INTERNAL: {type(e).__name__}: {e}", {"code": "E_INTERNAL"}

    def _handle(self, cap, action, args):
        if cap == "shell":
            cmd = args.get("command") or args.get("cmd") or ""
            if not cmd:
                raise ExecutorError("E_BAD_ARGS", "shell 需要 args.command")
            res = self._sandbox_run(cmd)
            extra = {"mode": res.mode, "exit_code": res.exit_code, "ms": round(res.ms, 1)}
            return bool(res.ok), res.output, ("" if res.ok else f"退出码 {res.exit_code}"), extra

        if cap == "read_file":
            p = self.policy.safe_path(args.get("path"))
            if not os.path.isfile(p):
                raise ExecutorError("E_NOT_FOUND", f"文件不存在：{p}")
            n = os.path.getsize(p)
            self.policy.check_size(n, "文件")
            with open(p, "rb") as f:
                raw = f.read()
            return True, base64.b64encode(raw).decode("ascii"), "", {"path": p, "bytes": n, "encoding": "base64"}

        if cap == "write_file":
            p = self.policy.safe_path(args.get("path"))
            content = args.get("content", "")
            raw = base64.b64decode(content) if args.get("encoding") == "base64" else str(content).encode("utf-8")
            self.policy.check_size(len(raw), "写入内容")
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "wb") as f:
                f.write(raw)
            return True, f"已写入 {len(raw)} 字节", "", {"path": p, "bytes": len(raw)}

        if cap == "list_dir":
            p = self.policy.safe_path(args.get("path") or ".")
            items = []
            for name in sorted(os.listdir(p))[:int(args.get("limit") or 200)]:
                full = os.path.join(p, name)
                items.append({"name": name, "dir": os.path.isdir(full),
                              "size": (os.path.getsize(full) if os.path.isfile(full) else 0)})
            return True, json.dumps(items, ensure_ascii=False), "", {"path": p, "count": len(items)}

        if cap == "screenshot":
            png = self.driver.screenshot()
            self.policy.check_size(len(png), "截图")
            extra = {"bytes": len(png), "format": "png", "driver": self.driver.kind,
                     "screen": list(self.driver.size())}
            if args.get("ocr_text"):
                extra["ocr_text"] = str(args["ocr_text"])[:8000]
            return True, base64.b64encode(png).decode("ascii"), "", extra

        if cap == "input":
            act = str(action or "").lower()
            if act == "click":
                self.driver.click(args.get("x"), args.get("y"), args.get("button", "left"), args.get("clicks", 1))
            elif act == "move":
                self.driver.move(args.get("x"), args.get("y"))
            elif act == "type":
                self.driver.type_text(args.get("text", ""), args.get("interval", 0.02))
            elif act == "key":
                self.driver.key(args.get("keys", ""))
            elif act == "scroll":
                self.driver.scroll(args.get("amount", 0))
            else:
                raise ExecutorError("E_BAD_ARGS", f"未知 input 动作 {act!r}（click/move/type/key/scroll）")
            return True, f"{act} ok", "", {"action": act, "driver": self.driver.kind}

        raise ExecutorError("E_UNKNOWN_CAP", f"未知能力 {cap!r}")

    # ---------- 主循环 ----------
    def run_command(self, payload):
        seq = payload.get("seq")
        cap, action = payload.get("cap"), payload.get("action")
        t0 = time.time()
        ok, output, error, extra = self.handle(cap, action, payload.get("args"),
                                               approved=payload.get("approved", False))
        ms = (time.time() - t0) * 1000
        self.stats["executed" if ok else "failed"] += 1
        self.local_audit.append({"t": round(time.time(), 3), "seq": seq, "cap": cap, "action": action,
                                 "ok": ok, "ms": round(ms, 1), "error": error[:200]})
        self.local_audit[:] = self.local_audit[-200:]
        result = {"device_id": self.device_id, "seq": seq, "ok": ok, "output": output,
                  "error": error, "extra": {**extra, "ms": round(ms, 1)}}
        if self.verbose:
            mark = "✅" if ok else "⛔"
            print(f"  [exec] {mark} {cap}:{action} {ms:.0f}ms {error or ''}", flush=True)
        try:
            self._post("/api/executor/result", result)
        except Exception as e:
            self.stats["errors"] += 1
            if self.verbose:
                print(f"  [exec] ⚠ 回结果失败：{e}", flush=True)
        return result

    def loop(self, duration=None, idle_timeout=None, max_commands=None):
        """主循环：注册 → 长轮询 → 执行 → 回结果（poll 同时起心跳作用）。"""
        if not self.registered:
            self.register()
        t0 = time.time()
        idle_since = time.time()
        served = 0
        while not self.stop_flag:
            if duration and (time.time() - t0) > duration:
                break
            if idle_timeout and (time.time() - idle_since) > idle_timeout:
                break
            if max_commands and served >= max_commands:
                break
            try:
                out = self._post("/api/executor/poll", {"device_id": self.device_id})
            except ExecutorError as e:
                if "E_HTTP_404" in e.code or "E_UNKNOWN_DEVICE" in e.code or "E_HTTP_401" in e.code:
                    if self.verbose:
                        print(f"  [exec] 中心不认识我（{e.message}），重新注册…", flush=True)
                    self.registered = False
                    time.sleep(1.0)
                    self.register()
                else:
                    self.stats["errors"] += 1
                    if self.verbose:
                        print(f"  [exec] ⚠ 轮询失败：{e.message}", flush=True)
                    time.sleep(2.0)
                continue
            self.stats["polled"] += 1
            cmds = out.get("commands") or []
            if not cmds:
                continue
            idle_since = time.time()
            for payload in cmds:
                self.run_command(payload)
                served += 1
        return dict(self.stats)
