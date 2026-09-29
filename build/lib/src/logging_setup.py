"""#7 完整日志（M5 尾巴）：结构化落盘 + 轮转 + 保留期 + 敏感信息脱敏。

M3 的 `src/trace.py` 解决「一次 run 内部每步看得见」（内存事件流 + JSONL 导出）；
本模块解决「部署后日志成体系」：

  · **结构化**：一条事件一行 JSON（`ts / level / event / run_id + 自定义字段`），
    落 `data/logs/forge-YYYYMMDD.jsonl`——既能人读，也能直接喂给日志采集/分析。
  · **轮转**：按天分文件；单文件超 `max_mb` 自动切分（`forge-YYYYMMDD.1.jsonl` …）。
  · **保留期**：超过 `keep_days` 的日志在启动/跨天时自动清理（不无限膨胀）。
  · **脱敏（部署服务的硬要求）**：字段名含 key/token/secret/authorization/password 的值、
    以及 `sk-xxx` / `Bearer xxx` / `gho_xxx` 形态的字符串，一律写成 `***`——**日志不能泄漏 API key**。
  · **级别过滤**：DEBUG/INFO/WARNING/ERROR，低于配置级别的直接丢（省磁盘、降噪音）。
  · **零依赖**：只用标准库（json/re/threading/pathlib），不引 logging 框架也能被 logging 接管。

配置（`config/models.yaml` 的 `logging` 段）：

    logging:
      enabled: true        # 关掉则完全不落盘（测试/临时排查用）
      level: INFO          # DEBUG / INFO / WARNING / ERROR
      dir: data/logs
      keep_days: 14        # 保留天数，超期自动清理
      max_mb: 20           # 单文件上限，超过切分
      console: false       # 是否同时打到 stderr（默认关，别干扰 CLI 的漂亮输出）

用法：

    from .logging_setup import get_logger
    log = get_logger()                       # 全局单例（读配置）
    log.info("run_start", run_id=rid, role="default", task="帮我算 1+1")
    sub = log.bind(run_id=rid, session=sid)  # 带上下文的子记录器
    sub.warning("tool_timeout", tool="run_command", sec=30)
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys
import threading

# ── 级别 ──────────────────────────────────────────────────────────
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

# ── 脱敏：字段名命中 → 整值打码；字符串命中 → 片段打码 ────────────────
_SENSITIVE_KEY_PARTS = ("key", "token", "secret", "authorization", "auth", "password", "passwd", "cookie")
_STRING_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),          # OpenAI 风格（DeepSeek/通义等同款）
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{6,}"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{10,}"),     # GitHub token
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),             # AWS access key id
]
_DEFAULT_CLIP = 2000  # 单字段最长保留字符数（防日志被大输出撑爆）


def _redact_str(s: str) -> str:
    for pat in _STRING_PATTERNS:
        s = pat.sub("***", s)
    return s


def redact(value, clip: int = _DEFAULT_CLIP):
    """递归脱敏 + 截断。dict 的敏感 key 整值打码，字符串按模式打码。"""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and any(p in k.lower() for p in _SENSITIVE_KEY_PARTS):
                out[k] = "***" if v else v
            else:
                out[k] = redact(v, clip=clip)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v, clip=clip) for v in value]
    if isinstance(value, str):
        s = _redact_str(value)
        return s if len(s) <= clip else s[:clip] + f"…(+{len(s) - clip}字)"
    return value


def _today() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")


def _ts() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


class Logger:
    """结构化日志器：JSONL 落盘 + 轮转 + 保留期 + 脱敏 + 级别过滤。

    线程安全（一把锁保护写入与轮转）；任何写入异常都不抛给调用方
    （日志挂了不能拖垮 Agent —— 只把 `last_error` 记下来）。
    """

    def __init__(self, log_dir=None, level="INFO", keep_days=14, max_mb=20,
                 console=False, enabled=True, context=None):
        self.dir = log_dir or os.path.join("data", "logs")
        self.level = str(level or "INFO").upper()
        self.keep_days = int(keep_days or 0)
        self.max_mb = float(max_mb or 0)
        self.console = bool(console)
        self.enabled = bool(enabled)
        self.context = dict(context or {})
        self.last_error = None
        self._lock = threading.Lock()
        self._pruned_on = None
        if self.enabled:
            self._ensure_dir()

    # ---------- 基础 ----------
    def _ensure_dir(self) -> bool:
        try:
            os.makedirs(self.dir, exist_ok=True)
            return True
        except Exception as e:                      # 目录不可写：记下原因，静默降级为不落盘
            self.last_error = f"日志目录不可用（{e}）"
            self.enabled = False
            return False

    def _path(self, seq: int = 0) -> str:
        name = f"forge-{_today()}.jsonl" if seq == 0 else f"forge-{_today()}.{seq}.jsonl"
        return os.path.join(self.dir, name)

    def _current_path(self) -> str:
        """当前应写入的文件：最后一个分片。"""
        seq = 0
        while os.path.exists(self._path(seq + 1)):
            seq += 1
        return self._path(seq)

    def _rotate_if_needed(self, path: str) -> str:
        if self.max_mb <= 0 or not os.path.exists(path):
            return path
        try:
            if os.path.getsize(path) < self.max_mb * 1024 * 1024:
                return path
        except OSError:
            return path
        seq = 1
        while os.path.exists(self._path(seq)):
            seq += 1
        return self._path(seq)

    # ---------- 保留期 ----------
    def prune(self) -> int:
        """清理超过 keep_days 的日志分片，返回删除数。"""
        if not self.enabled or self.keep_days <= 0 or not os.path.isdir(self.dir):
            return 0
        cutoff = datetime.datetime.now() - datetime.timedelta(days=self.keep_days)
        removed = 0
        for name in os.listdir(self.dir):
            if not (name.startswith("forge-") and ".jsonl" in name):
                continue
            p = os.path.join(self.dir, name)
            try:
                if datetime.datetime.fromtimestamp(os.path.getmtime(p)) < cutoff:
                    os.remove(p)
                    removed += 1
            except OSError:
                continue
        return removed

    # ---------- 写入 ----------
    def log(self, level: str, event: str, **fields):
        """记一条事件。level 低于配置级别 → 丢弃。永不抛异常。"""
        if not self.enabled:
            return
        lv = str(level).upper()
        if _LEVELS.get(lv, 20) < _LEVELS.get(self.level, 20):
            return
        record = {"ts": _ts(), "level": lv, "event": event}
        record.update(self.context)
        record.update(fields)
        record = redact(record)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            try:
                if self._pruned_on != _today():
                    self.prune()
                    self._pruned_on = _today()
                path = self._rotate_if_needed(self._current_path())
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                self.last_error = str(e)
        if self.console:
            try:
                print(f"  [{lv}] {event} " + json.dumps(redact(fields), ensure_ascii=False),
                      file=sys.stderr, flush=True)
            except Exception:
                pass

    def debug(self, event, **fields):
        self.log("DEBUG", event, **fields)

    def info(self, event, **fields):
        self.log("INFO", event, **fields)

    def warning(self, event, **fields):
        self.log("WARNING", event, **fields)

    def error(self, event, **fields):
        self.log("ERROR", event, **fields)

    def bind(self, **fields) -> "Logger":
        """派生一个带固定上下文的子日志器（共享同一落盘目标）。"""
        child = Logger(log_dir=self.dir, level=self.level, keep_days=self.keep_days,
                       max_mb=self.max_mb, console=self.console, enabled=self.enabled,
                       context={**self.context, **fields})
        child._lock = self._lock                 # 共享锁与状态
        child.last_error = self.last_error
        return child

    # ---------- 读取 / 管理（/logs 命令用）----------
    def files(self) -> list:
        if not os.path.isdir(self.dir):
            return []
        out = [os.path.join(self.dir, n) for n in os.listdir(self.dir)
               if n.startswith("forge-") and ".jsonl" in n]
        return sorted(out)

    def read(self, limit=50, level=None) -> list:
        """读最近 limit 条（可选按最低级别过滤），返回 dict 列表。"""
        rows = []
        for path in self.files():
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if level and _LEVELS.get(str(rec.get("level", "")).upper(), 0) < _LEVELS.get(level, 0):
                            continue
                        rows.append(rec)
            except OSError:
                continue
        return rows[-int(limit):] if limit else rows

    def errors(self, limit=20) -> list:
        return self.read(limit=limit, level="WARNING")

    def stats(self) -> dict:
        files = self.files()
        size = 0
        for p in files:
            try:
                size += os.path.getsize(p)
            except OSError:
                pass
        return {"enabled": self.enabled, "dir": os.path.abspath(self.dir), "files": len(files),
                "bytes": size, "level": self.level, "keep_days": self.keep_days,
                "max_mb": self.max_mb, "last_error": self.last_error}

    def clear(self, level=None) -> int:
        """清空日志：level 为空删全部文件，否则只删含该级别以上记录的文件。"""
        removed = 0
        for p in self.files():
            try:
                if level is None:
                    os.remove(p)
                    removed += 1
                else:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        if any(level in l for l in f):
                            os.remove(p)
                            removed += 1
            except OSError:
                continue
        return removed


# ── 全局单例（配置驱动，同 memory 的用法）──────────────────────────
_LOGGER = None


def init_logger(cfg) -> Logger:
    """按配置初始化单例；配置里没有 logging 段就用默认值。"""
    global _LOGGER
    conf = (cfg or {}).get("logging") or {}
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = conf.get("dir") or os.path.join("data", "logs")
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(base_dir, log_dir)
    _LOGGER = Logger(log_dir=log_dir, level=conf.get("level", "INFO"),
                     keep_days=conf.get("keep_days", 14), max_mb=conf.get("max_mb", 20),
                     console=conf.get("console", False), enabled=conf.get("enabled", True))
    return _LOGGER


def get_logger(cfg=None) -> Logger:
    """取全局日志器（首次调用时按配置初始化）。"""
    global _LOGGER
    if _LOGGER is None:
        if cfg is None:
            try:
                from .config import load_config

                cfg = load_config()
            except Exception:
                cfg = {}
        init_logger(cfg)
    return _LOGGER


def reset_logger():
    """重置单例（配置热重载/测试用）。"""
    global _LOGGER
    _LOGGER = None


def log_event(level, event, **fields):
    """便捷入口：用全局单例记一条（任何异常都吞掉，日志永不拖垮主流程）。"""
    try:
        get_logger().log(level, event, **fields)
    except Exception:
        pass
