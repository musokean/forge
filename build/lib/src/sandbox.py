"""#4 工具安全沙箱（M5 尾巴）：把 `run_command` 的不可信执行隔离开（A06）。

四层策略（`config/models.yaml` 的 `sandbox.mode`）：

    auto    （默认）Docker 可用 → 容器隔离；不可用 → **加固的本机执行**（并明确提示降级）
    docker  强制容器；Docker 不可用 → **拒绝执行**（不静默降级，生产部署用这个）
    local   加固的本机执行（不依赖 Docker）
    off     直通（等同旧行为，仅本地调试用；会记一条 WARNING 日志）

**容器约束**（`docker run` 参数由测试逐条断言）：

    --rm --network=none            一次性；默认**断网**（sandbox.network: true 才放网）
    --memory/--cpus/--pids-limit   资源上限，防 fork 炸弹/吃光内存
    --read-only + tmpfs /tmp       根文件系统只读，只给 /tmp 一点可写空间
    --user 65534:65534             非 root 运行
    -v <workdir>:/work:ro -w /work 工作目录**只读**挂载（sandbox.mount_rw: true 才可写）
    -e 仅白名单环境变量            宿主环境变量不整体透传（否则 API key 会被命令读走）
    + 超时后 docker rm -f <name>   杀死残留容器

**本地加固**（Docker 不可用时的降级，仍有实际价值）：

    · 环境变量白名单：只透传 PATH/SYSTEMROOT/TEMP 等必需项——**不再把 os.environ 整体塞给子进程**
      （旧实现会把 DEEPSEEK_API_KEY 这类宿主密钥暴露给任意命令）
    · 危险命令模式拦截（rm -rf /、mkfs、dd 写盘、shutdown、format…，可配置追加）
    · 超时强杀 + 输出截断（默认 10 万字符）

> 能力边界：本模块挡的是「命令执行的副作用与密钥泄漏」；**审批层（#4 审批）挡的是「要不要执行」**，
> 两层叠加才是 A06 的完整防线——沙箱不是审批的替代品。
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import uuid

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_OUTPUT = 100_000
DEFAULT_MEMORY = "256m"
DEFAULT_CPUS = "1.0"
DEFAULT_PIDS = 128

# 危险命令模式（默认拦截；config sandbox.deny_patterns 可追加）
DENY_PATTERNS = [
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/(\s|$)",     # rm -rf /
    r":\(\)\s*\{.*\}\s*;",                        # fork 炸弹
    r"\bmkfs(\.\w+)?\b",                          # 格式化文件系统
    r"\bdd\s+.*of=/dev/(sd|nvme|hd)",             # 写裸设备
    r"\b(shutdown|reboot|halt|poweroff)\b",
    r"\bformat\s+[a-zA-Z]:",                      # Windows 格式化
    r"\bdiskpart\b",
    r">\s*/dev/(sd|nvme)",
    r"\bchmod\s+-R\s+777\s+/(\s|$)",
]

# 透传给子进程的环境变量白名单（本地执行用；容器只透传这里挑出的非敏感项）
ENV_WHITELIST = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "LANG", "LC_ALL",
    "PYTHONIOENCODING", "PYTHONUTF8", "NUMBER_OF_PROCESSORS",
)


class SandboxUnavailable(RuntimeError):
    """需要 Docker 但 Docker 不可用（mode=docker 时直接拒绝执行）。"""


class SandboxResult:
    """一次沙箱执行的结果。"""

    def __init__(self, ok, output, exit_code=None, mode="local", ms=0.0, error="", command=""):
        self.ok = ok
        self.output = output
        self.exit_code = exit_code
        self.mode = mode
        self.ms = ms
        self.error = error
        self.command = command

    def as_dict(self):
        return {"ok": self.ok, "mode": self.mode, "exit_code": self.exit_code,
                "ms": round(self.ms, 1), "error": self.error}

    def __repr__(self):
        return f"<SandboxResult ok={self.ok} mode={self.mode} code={self.exit_code} {self.ms:.0f}ms>"


# ── Docker 探测（带缓存，避免每条命令都 shell 一次 docker）────────────
_DOCKER_CACHE = {"at": 0.0, "available": None, "version": None}
_DOCKER_TTL = 60.0


def docker_available(refresh=False, timeout=6.0) -> bool:
    """本机 Docker 是否可用（`docker version` 能拿到 Server 版本才算）。"""
    now = time.time()
    if not refresh and _DOCKER_CACHE["available"] is not None and now - _DOCKER_CACHE["at"] < _DOCKER_TTL:
        return _DOCKER_CACHE["available"]
    available, version = False, None
    try:
        r = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                           capture_output=True, timeout=timeout, text=True)
        if r.returncode == 0 and (r.stdout or "").strip():
            available, version = True, r.stdout.strip()
    except Exception:
        available, version = False, None
    _DOCKER_CACHE.update({"at": now, "available": available, "version": version})
    return available


def docker_version():
    docker_available()
    return _DOCKER_CACHE.get("version")


def _reset_docker_cache():
    """测试用：清掉探测缓存。"""
    _DOCKER_CACHE.update({"at": 0.0, "available": None, "version": None})


def _clip(text, limit=DEFAULT_MAX_OUTPUT):
    text = text or ""
    if limit and len(text) > limit:
        return text[:limit] + f"\n…（输出过长已截断，共 {len(text)} 字符）"
    return text


def _decode(b):
    if isinstance(b, str):
        return b
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return (b or b"").decode("utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════
# 本地加固执行
# ══════════════════════════════════════════════════════════════════
class LocalSandbox:
    """加固的本机执行：环境变量白名单 + 危险模式拦截 + 超时 + 输出截断。"""

    mode = "local"

    def __init__(self, timeout=DEFAULT_TIMEOUT, max_output=DEFAULT_MAX_OUTPUT,
                 deny_patterns=None, extra_env=None, cwd=None):
        self.timeout = int(timeout or DEFAULT_TIMEOUT)
        self.max_output = int(max_output or DEFAULT_MAX_OUTPUT)
        self.deny = [re.compile(p, re.IGNORECASE) for p in (DENY_PATTERNS + list(deny_patterns or []))]
        self.extra_env = dict(extra_env or {})
        self.cwd = cwd

    def denied(self, command: str):
        """命中危险模式则返回该模式，否则 None。"""
        for pat in self.deny:
            if pat.search(command or ""):
                return pat.pattern
        return None

    def safe_env(self) -> dict:
        """只透传白名单里的环境变量（关键：不把宿主密钥整体交给子进程）。"""
        env = {k: os.environ[k] for k in ENV_WHITELIST if k in os.environ}
        env.update(self.extra_env)
        env.setdefault("FORGE_SANDBOX", "local")
        return env

    def run(self, command: str, timeout=None, cwd=None) -> SandboxResult:
        t0 = time.time()
        hit = self.denied(command)
        if hit:
            return SandboxResult(False, f"⛔ 命令被沙箱拦截（匹配危险模式：{hit}）", None,
                                 self.mode, (time.time() - t0) * 1000,
                                 error="denied", command=command)
        try:
            r = subprocess.run(command, shell=True, capture_output=True,
                               timeout=timeout or self.timeout,
                               env=self.safe_env(), cwd=cwd or self.cwd)
            out = _decode(r.stdout) + _decode(r.stderr)
            ok = r.returncode == 0
            return SandboxResult(ok, _clip(out.strip() or "(命令无输出)", self.max_output),
                                 r.returncode, self.mode, (time.time() - t0) * 1000, command=command)
        except subprocess.TimeoutExpired:
            return SandboxResult(False, f"命令超时（{timeout or self.timeout} 秒，已强杀）", None,
                                 self.mode, (time.time() - t0) * 1000, error="timeout", command=command)
        except Exception as e:
            return SandboxResult(False, f"命令执行失败：{e}", None, self.mode,
                                 (time.time() - t0) * 1000, error=str(e), command=command)


# ══════════════════════════════════════════════════════════════════
# Docker 容器隔离
# ══════════════════════════════════════════════════════════════════
class DockerSandbox:
    """容器隔离执行：断网 + 资源限制 + 只读挂载 + 非 root + 超时清容器。"""

    mode = "docker"

    def __init__(self, image=DEFAULT_IMAGE, timeout=DEFAULT_TIMEOUT, network=False,
                 memory=DEFAULT_MEMORY, cpus=DEFAULT_CPUS, pids_limit=DEFAULT_PIDS,
                 workdir=None, mount_rw=False, read_only_root=True,
                 max_output=DEFAULT_MAX_OUTPUT, deny_patterns=None, env_passthrough=None,
                 runner=None):
        self.image = image or DEFAULT_IMAGE
        self.timeout = int(timeout or DEFAULT_TIMEOUT)
        self.network = bool(network)
        self.memory = str(memory or DEFAULT_MEMORY)
        self.cpus = str(cpus or DEFAULT_CPUS)
        self.pids_limit = int(pids_limit or DEFAULT_PIDS)
        self.workdir = workdir or _BASE_DIR
        self.mount_rw = bool(mount_rw)
        self.read_only_root = bool(read_only_root)
        self.max_output = int(max_output or DEFAULT_MAX_OUTPUT)
        self.deny = [re.compile(p, re.IGNORECASE) for p in (DENY_PATTERNS + list(deny_patterns or []))]
        self.env_passthrough = list(env_passthrough or [])
        self._runner = runner or subprocess.run   # 测试注入点

    def denied(self, command: str):
        for pat in self.deny:
            if pat.search(command or ""):
                return pat.pattern
        return None

    def build_argv(self, command: str, name: str, cwd=None):
        """构造 docker run 参数（测试逐条断言的就是这里）。"""
        mount = cwd or self.workdir
        argv = ["docker", "run", "--rm",
                "--name", name,
                "--network", "bridge" if self.network else "none",
                "--memory", self.memory,
                "--cpus", self.cpus,
                "--pids-limit", str(self.pids_limit),
                "--user", "65534:65534",
                ]
        if self.read_only_root:
            argv += ["--read-only", "--tmpfs", "/tmp:rw,size=64m"]
        argv += ["-v", f"{mount}:/work" + ("" if self.mount_rw else ":ro"), "-w", "/work"]
        for k in self.env_passthrough:
            if k in os.environ:
                argv += ["-e", f"{k}={os.environ[k]}"]
        argv += [self.image, "sh", "-c", command]
        return argv

    def run(self, command: str, timeout=None, cwd=None) -> SandboxResult:
        t0 = time.time()
        hit = self.denied(command)
        if hit:
            return SandboxResult(False, f"⛔ 命令被沙箱拦截（匹配危险模式：{hit}）", None,
                                 self.mode, (time.time() - t0) * 1000,
                                 error="denied", command=command)
        if not docker_available():
            raise SandboxUnavailable(
                "需要 Docker 但本机不可用（docker version 失败）。"
                "装 Docker Desktop 后重试，或把 config/models.yaml 的 sandbox.mode 设为 local/auto。")
        name = f"forge-sbx-{uuid.uuid4().hex[:10]}"
        argv = self.build_argv(command, name, cwd=cwd)
        try:
            r = self._runner(argv, capture_output=True, timeout=timeout or self.timeout)
            out = _decode(r.stdout) + _decode(r.stderr)
            ok = r.returncode == 0
            return SandboxResult(ok, _clip(out.strip() or "(命令无输出)", self.max_output),
                                 r.returncode, self.mode, (time.time() - t0) * 1000, command=command)
        except subprocess.TimeoutExpired:
            self._kill_container(name)
            return SandboxResult(False, f"命令超时（{timeout or self.timeout} 秒，已删除容器）", None,
                                 self.mode, (time.time() - t0) * 1000, error="timeout", command=command)
        except SandboxUnavailable:
            raise
        except Exception as e:
            return SandboxResult(False, f"容器执行失败：{e}", None, self.mode,
                                 (time.time() - t0) * 1000, error=str(e), command=command)

    def _kill_container(self, name):
        """超时后清掉残留容器（best-effort）。"""
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# 门面：按策略选执行器
# ══════════════════════════════════════════════════════════════════
class Sandbox:
    """统一入口：按 `sandbox.mode` 决定用容器还是加固本机执行。"""

    def __init__(self, cfg=None, mode=None, **kw):
        # 注意：mode 默认必须是 None——写成 "auto" 会因为 truthy 把配置里的 mode 覆盖掉
        # （2026-09-28 自测揪出：sandbox.mode=docker/off 全被当成 auto，隔离策略静默失效）
        conf = ((cfg or {}).get("sandbox") or {}) if isinstance(cfg, dict) else {}
        self.mode = str(mode or conf.get("mode") or "auto").lower()
        self.timeout = kw.get("timeout") or conf.get("timeout") or DEFAULT_TIMEOUT
        self.max_output = kw.get("max_output") or conf.get("max_output") or DEFAULT_MAX_OUTPUT
        self.deny_patterns = list(conf.get("deny_patterns") or []) + list(kw.get("deny_patterns") or [])
        self.image = conf.get("image") or DEFAULT_IMAGE
        self.network = conf.get("network", False)
        self.memory = conf.get("memory") or DEFAULT_MEMORY
        self.cpus = conf.get("cpus") or DEFAULT_CPUS
        self.pids_limit = conf.get("pids_limit") or DEFAULT_PIDS
        self.mount_rw = conf.get("mount_rw", False)
        self.workdir = conf.get("workdir") or kw.get("workdir") or _BASE_DIR
        self.env_passthrough = conf.get("env_passthrough") or []
        self.docker_runner = kw.get("docker_runner")     # 测试注入：伪造 docker 调用
        self._docker = None
        self._local = None

    # ---- 执行器 ----
    @property
    def docker(self) -> DockerSandbox:
        if self._docker is None:
            self._docker = DockerSandbox(
                image=self.image, timeout=self.timeout, network=self.network, memory=self.memory,
                cpus=self.cpus, pids_limit=self.pids_limit, workdir=self.workdir,
                mount_rw=self.mount_rw, max_output=self.max_output, deny_patterns=self.deny_patterns,
                env_passthrough=self.env_passthrough, runner=self.docker_runner)
        return self._docker

    @property
    def local(self) -> LocalSandbox:
        if self._local is None:
            self._local = LocalSandbox(timeout=self.timeout, max_output=self.max_output,
                                       deny_patterns=self.deny_patterns, cwd=self.workdir)
        return self._local

    def resolve_mode(self) -> str:
        """把 auto 解析成实际生效的执行器（docker / local）。"""
        if self.mode == "auto":
            return "docker" if docker_available() else "local"
        return self.mode

    def _log(self, level, event, **fields):
        try:
            from .logging_setup import log_event

            log_event(level, event, **fields)
        except Exception:
            pass

    def run(self, command: str, timeout=None, cwd=None) -> SandboxResult:
        """执行命令：返回 SandboxResult（不抛业务异常；Docker 缺失且强制 docker 时抛）。"""
        mode = self.resolve_mode()
        if mode == "docker":
            if not docker_available():
                raise SandboxUnavailable(
                    "sandbox.mode=docker 但本机 Docker 不可用——为安全起见拒绝在本机直接执行。"
                    "装 Docker Desktop，或把 mode 改成 auto/local。")
            res = self.docker.run(command, timeout=timeout, cwd=cwd)
        elif mode == "off":
            self._log("WARNING", "sandbox_off", command=command)
            res = LocalSandbox(timeout=self.timeout, max_output=self.max_output, cwd=self.workdir,
                               deny_patterns=[]).run(command, timeout=timeout, cwd=cwd)
            res.mode = "off"
        else:
            res = self.local.run(command, timeout=timeout, cwd=cwd)
        self._log("INFO" if res.ok else "WARNING", "sandbox_run", mode=res.mode, command=command,
                  exit_code=res.exit_code, ms=res.ms, error=res.error)
        return res

    def status(self) -> dict:
        """给 `/sandbox` 命令看的状态。"""
        resolved = self.resolve_mode()
        return {
            "configured_mode": self.mode,
            "effective_mode": resolved,
            "docker_available": docker_available(),
            "docker_version": docker_version(),
            "image": self.image,
            "limits": {"network": self.network, "memory": self.memory, "cpus": self.cpus,
                       "pids": self.pids_limit, "mount_rw": self.mount_rw, "timeout": self.timeout,
                       "workdir": self.workdir},
            "deny_patterns": len(DENY_PATTERNS) + len(self.deny_patterns),
        }


# ── 全局单例（配置驱动）──────────────────────────────────────────
_SANDBOX = None
_LOCK = threading.Lock()


def get_sandbox(cfg=None, refresh=False) -> Sandbox:
    """取全局沙箱（首次调用按配置初始化；refresh=True 重新读配置）。"""
    global _SANDBOX
    with _LOCK:
        if _SANDBOX is None or refresh:
            if cfg is None:                    # refresh 也要重读配置（否则热生效会退回默认值）
                try:
                    from .config import load_config

                    cfg = load_config()
                except Exception:
                    cfg = {}
            _SANDBOX = Sandbox(cfg)
        return _SANDBOX


def reset_sandbox():
    global _SANDBOX
    _SANDBOX = None
