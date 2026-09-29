"""#14 部署与服务化：把 forge 变成可被程序调用的 API 服务（FastAPI）。

设计对齐 A12（部署与服务化）：

  · 可选依赖——fastapi / uvicorn 只在服务化场景需要，核心仍是零重依赖：
        pip install "handcraft-agent[server]"
    没装时 import 不报错，只有真正 create_app()/serve() 才提示装哪个包。
  · 会话——#12 的 Web 是单会话；服务化必须多会话：SQLite 持久化到 data/sessions.db，
    每个会话独立 Agent 上下文，断了能续聊、换个客户端接着聊。
  · 鉴权——Authorization: Bearer <key> 或 X-API-Key: <key>；key 来自配置 server.api_keys
    或环境变量 FORGE_API_KEY（逗号分隔），支持 env:变量名 间接引用。
    **未配 key 时自动退化为「仅本机回环可访问」**：本地开发零配置，对外服务必须配 key。
  · 限流——按调用方滑动窗口（server.rate_limit_per_min，默认 60），超过返回 429 + Retry-After。
  · 写操作——服务端没有交互审批通道 → 默认拒绝（只读能力全可用），与 #12 Web 一致；
    审批模式可用 server.approve_mode 覆盖（callback/auto_approve 仅供受控部署）。
  · 可观测——每请求一行访问日志（方法 / 路径 / 状态 / 耗时 / 客户端 / 会话）。

启动：
    forge --serve [--host 127.0.0.1] [--port 8080]        # 或 REPL 里 /serve
    curl -H "Authorization: Bearer $KEY" -X POST http://127.0.0.1:8080/api/chat \
         -H "Content-Type: application/json" -d '{"message": "你好"}'

接口一览（详见 GET /docs 的 OpenAPI 页面）：
    GET    /healthz                 探活（无鉴权，供负载均衡/容器健康检查）
    GET    /api/status              服务状态（模型 / 会话数 / 鉴权模式 / 限流）
    POST   /api/chat                {"message": "...", "session_id": "可选"} → 回复 + usage
    POST   /api/sessions            新建会话 {"title": "可选"} → {"session_id": ...}
    GET    /api/sessions            会话列表（含轮数、最后活跃时间）
    GET    /api/sessions/{sid}      会话详情 + 完整消息历史
    DELETE /api/sessions/{sid}      删除会话
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

# ── 可选依赖：缺失时模块仍可 import（纯逻辑/测试不需要 fastapi）─────────────
try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse

    HAS_FASTAPI = True
except Exception:  # pragma: no cover - 取决于环境是否装了可选依赖
    HAS_FASTAPI = False

try:
    import uvicorn

    HAS_UVICORN = True
except Exception:  # pragma: no cover
    HAS_UVICORN = False

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
__version__ = "0.4.0"  # 与 pyproject.toml 保持一致

MISSING_DEPS_HINT = (
    '缺少服务化依赖：请先安装 —— pip install "handcraft-agent[server]"'
    "（或 pip install fastapi uvicorn）"
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_RATE_LIMIT = 60  # 次/分/调用方；0 = 关闭限流
_LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}


# ══════════════════════════════════════════════════════════════════
# 鉴权 / 限流（纯函数 + 小对象，不依赖 fastapi，可单独测试）
# ══════════════════════════════════════════════════════════════════
def is_loopback(host: str) -> bool:
    """判断来源是否本机（无 key 模式下的白名单）。"""
    host = (host or "").strip()
    return host in _LOOPBACK or host.startswith("127.")


def resolve_api_keys(cfg) -> list:
    """从配置 + 环境变量解析可用 key 列表。

    取值顺序：config server.api_keys（字符串或列表）→ server.api_key → 环境变量 FORGE_API_KEY（逗号分隔）。
    每项支持 `env:变量名` 间接引用（key 不落配置文件，适合容器/CI）。
    """
    server = (cfg or {}).get("server") or {}
    raw = server.get("api_keys", server.get("api_key"))
    items = []
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        items = [str(p).strip() for p in raw]
    env_raw = os.environ.get("FORGE_API_KEY", "")
    items += [p.strip() for p in env_raw.split(",")]

    keys = []
    for it in items:
        if not it:
            continue
        if it.startswith("env:"):
            it = os.environ.get(it[4:], "")
        it = (it or "").strip()
        if it and it not in keys:
            keys.append(it)
    return keys


def check_auth(keys, provided, client_host="") -> tuple:
    """判定是否放行，返回 (allow, reason)。

    · 配了 key：必须提供且匹配（常量时间比较，防时序侧信道）；
    · 没配 key：只放行本机回环——对外服务必须配 key，避免「裸奔的 Agent」。
    """
    if not keys:
        if is_loopback(client_host):
            return True, "本机访问（未配置 API key）"
        return False, "服务未配置 API key，仅允许本机访问；对外服务请配置 server.api_keys"
    if not provided:
        return False, "缺少 API key（请用 Authorization: Bearer <key> 或 X-API-Key 头）"
    for k in keys:
        if hmac.compare_digest(str(provided), str(k)):
            return True, "API key 校验通过"
    return False, "API key 无效"


def extract_key(authorization=None, x_api_key=None) -> str:
    """从两种常见头部取出 key（Bearer 优先，兼容裸 token）。"""
    if x_api_key:
        return str(x_api_key).strip()
    if authorization:
        parts = str(authorization).split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return str(authorization).strip()
    return ""


def identity_of(keys, provided, client_host="") -> str:
    """调用方标识（限流用）：有 key 用 key 指纹，没 key 用来源 IP。"""
    if keys and provided:
        return "key:" + hashlib.sha256(str(provided).encode("utf-8")).hexdigest()[:12]
    return "ip:" + (client_host or "unknown")


class RateLimiter:
    """按标识的滑动窗口限流（线程安全；limit<=0 表示关闭）。"""

    def __init__(self, limit=DEFAULT_RATE_LIMIT, window=60.0):
        self.limit = int(limit or 0)
        self.window = float(window)
        self._hits = {}
        self._lock = threading.Lock()

    def hit(self, ident: str, now=None) -> float:
        """记一次调用；返回需等待秒数（0 = 放行）。"""
        if self.limit <= 0:
            return 0.0
        now = time.time() if now is None else now
        with self._lock:
            q = [t for t in self._hits.get(ident, []) if now - t < self.window]
            if len(q) >= self.limit:
                self._hits[ident] = q
                return max(0.0, self.window - (now - q[0]))
            q.append(now)
            self._hits[ident] = q
        return 0.0

    def reset(self, ident=None):
        with self._lock:
            if ident is None:
                self._hits.clear()
            else:
                self._hits.pop(ident, None)


# ══════════════════════════════════════════════════════════════════
# 会话存储（SQLite 持久化；journal_mode=MEMORY 沿用 knowledge/tasks 的沙箱适配）
# ══════════════════════════════════════════════════════════════════
def _strip_injected(msg, injected):
    """把当次注入的记忆前缀从历史消息里剥掉（只处理 user 角色，其它原样）。"""
    if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
        return msg
    content = msg["content"]
    if not injected or not content.startswith(injected):
        return msg
    content = content[len(injected):].lstrip("\n")
    if content.startswith("用户问题："):
        content = content[len("用户问题："):]
    out = dict(msg)
    out["content"] = content
    return out


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class SessionStore:
    """会话 + 消息历史持久化。消息按 JSON 原样存，保留 tool_calls 等结构。"""

    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(_BASE_DIR, "data", "sessions.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # 沙箱/网络盘上 DELETE journal 的锁语义会挂起（knowledge.py 同款处理）
            conn.execute("PRAGMA journal_mode=MEMORY")
            conn.execute("PRAGMA synchronous=OFF")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init(self):
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                " id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT, turns INTEGER DEFAULT 0)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, seq INTEGER, data TEXT)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_msg_sid ON messages(session_id, seq)")

    # ---- 会话 ----
    def create(self, title=None, session_id=None) -> str:
        sid = session_id or hashlib.sha256(
            f"{time.time()}-{os.urandom(8).hex()}".encode("utf-8")
        ).hexdigest()[:12]
        ts = _now()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO sessions (id, title, created_at, updated_at, turns)"
                " VALUES (?,?,?,?, COALESCE((SELECT turns FROM sessions WHERE id=?), 0))",
                (sid, (title or "新会话")[:80], ts, ts, sid),
            )
        return sid

    def get(self, sid) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
            if not r:
                return {}
            n = c.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (sid,)).fetchone()[0]
        d = dict(r)
        d["messages"] = int(n)
        return d

    def list_sessions(self, limit=50) -> list:
        with self._conn() as c:
            rows = c.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS messages"
                " FROM sessions s ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, sid) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM sessions WHERE id=?", (sid,))
            c.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            return cur.rowcount > 0

    def count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])

    # ---- 消息 ----
    def save_messages(self, sid, messages, turns=None):
        """整段覆盖保存（幂等：每次对话后按 Agent 当前历史重写，避免增量对不齐）。"""
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            for i, m in enumerate(messages):
                c.execute(
                    "INSERT INTO messages (session_id, seq, data) VALUES (?,?,?)",
                    (sid, i, json.dumps(m, ensure_ascii=False, default=str)),
                )
            if turns is None:
                turns = sum(1 for m in messages if m.get("role") == "user")
            c.execute("UPDATE sessions SET updated_at=?, turns=? WHERE id=?", (_now(), int(turns), sid))

    def history(self, sid, limit=200) -> list:
        with self._conn() as c:
            rows = c.execute(
                "SELECT data FROM messages WHERE session_id=? ORDER BY seq ASC LIMIT ?",
                (sid, int(limit)),
            ).fetchall()
        out = []
        for r in rows:
            try:
                out.append(json.loads(r["data"]))
            except Exception:
                continue
        return out

    def touch_title(self, sid, title):
        if not title:
            return
        with self._conn() as c:
            c.execute("UPDATE sessions SET title=? WHERE id=? AND (title IS NULL OR title='新会话')",
                      (str(title)[:80], sid))


# ══════════════════════════════════════════════════════════════════
# 服务层：会话 → Agent 缓存 → 对话
# ══════════════════════════════════════════════════════════════════
class ForgeService:
    """把 Agent 包装成「按会话隔离 + 可持久化 + 可注入假 Agent（测试）」的服务。"""

    def __init__(self, cfg=None, store=None, agent_factory=None, approve_mode="auto_reject",
                 max_cached_agents=64):
        from .config import load_config

        self.cfg = cfg or load_config()
        server = (self.cfg.get("server") or {}) if isinstance(self.cfg, dict) else {}
        self.store = store or SessionStore(server.get("db_path"))
        self.approve_mode = approve_mode
        self.max_cached_agents = int(max_cached_agents or 0)
        self._factory = agent_factory or self._default_factory
        self._agents = {}
        self._locks = {}
        self._guard = threading.Lock()
        self._warned = set()

    # ---- Agent 工厂 ----
    def _default_factory(self, session_id):
        """真实 Agent：服务端无交互审批 → 默认拒绝写操作（与 #12 Web 一致）。"""
        from .agent import Agent
        from .approval import Approver

        return Agent(stream=False, approver=Approver(mode=self.approve_mode), show_spinner=False)

    def _lock(self, sid):
        with self._guard:
            return self._locks.setdefault(sid, threading.Lock())

    def forget(self, sid):
        with self._guard:
            self._agents.pop(sid, None)

    # ---- 会话 ----
    def create_session(self, title=None, session_id=None):
        return self.store.create(title=title, session_id=session_id)

    def agent_for(self, sid):
        """取会话的 Agent（懒建 + 恢复历史）；同一会话串行执行。"""
        agent = self._agents.get(sid)
        if agent is not None:
            return agent
        agent = self._factory(sid)
        hist = self.store.history(sid)
        msgs = getattr(agent, "messages", None)
        if hist and isinstance(msgs, list):
            system = msgs[0] if msgs and msgs[0].get("role") == "system" else None
            agent.messages = ([system] if system else []) + hist
        if self.max_cached_agents and len(self._agents) >= self.max_cached_agents:
            self._agents.pop(next(iter(self._agents)), None)  # 简单 FIFO 淘汰，防内存无界
        with self._guard:
            self._agents[sid] = agent
        return agent

    def _with_memory(self, message):
        """长期记忆钩子（与 CLI / Web 一致：先自动沉淀，再召回注入）。

        返回 (发给模型的文本, 注入的记忆前缀)。前缀只在**当次请求**生效，
        不落库——否则 GET /api/sessions/{sid} 读出来的历史会被注入文本糊住（2026-09-28 踩到）。
        """
        try:
            from .memory import get_memory

            mem = get_memory()
            mem.auto_remember(message)
            ctx = mem.compose_context(message)
            if ctx:
                return ctx + "\n\n用户问题：" + message, ctx
        except Exception:
            pass
        return message, ""

    def _warning(self, reply):
        if "用户拒绝了该写操作" in (reply or "") and "write" not in self._warned:
            self._warned.add("write")
            return ("提示：服务端写操作默认被拒（无交互审批通道），请回 CLI（forge）执行；"
                    "受控部署可用 config server.approve_mode 调整。")
        return ""

    def _usage(self, agent):
        tok = getattr(agent, "total_tokens", None) or {}
        ctx = None
        try:
            ctx = agent._context_tokens()
        except Exception:
            pass
        budget = getattr(agent, "max_context_tokens", None)
        return {
            "prompt": int(tok.get("prompt", 0) or 0),
            "completion": int(tok.get("completion", 0) or 0),
            "context_tokens": ctx,
            "context_pct": round(100.0 * ctx / budget, 1) if (ctx and budget) else None,
        }

    async def chat(self, message, session_id=None):
        """跑一轮对话：返回值直接作为 /api/chat 响应体。"""
        message = (message or "").strip()
        if not message:
            raise ValueError("message 不能为空")
        sid = session_id or self.create_session(title=message[:40])
        lock = self._lock(sid)
        lock.acquire()
        try:
            agent = self.agent_for(sid)
            prompt, injected = self._with_memory(message)
            reply = await agent.run(prompt)
            saved = [m for m in getattr(agent, "messages", []) if m.get("role") != "system"]
            if injected:
                saved = [_strip_injected(m, injected) for m in saved]
            self.store.save_messages(sid, saved)
            self.store.touch_title(sid, message[:40])
            out = {"session_id": sid, "reply": reply}
            warn = self._warning(reply)
            if warn:
                out["warning"] = warn
            out["usage"] = self._usage(agent)
            return out
        finally:
            lock.release()

    def status(self, keys=None, rate_limit=None):
        try:
            from .config import resolve_model

            model = resolve_model(self.cfg, "default").get("model")
        except Exception:
            model = "?"
        return {
            "name": "forge",
            "version": __version__,
            "model": model,
            "sessions": self.store.count(),
            "active_agents": len(self._agents),
            "auth": "api_key" if keys else "loopback-only",
            "rate_limit_per_min": int(rate_limit or 0),
            "approve_mode": self.approve_mode,
        }


# ══════════════════════════════════════════════════════════════════
# FastAPI 应用
# ══════════════════════════════════════════════════════════════════
def create_app(cfg=None, service=None, api_keys=None, rate_limit=None, store=None, hub=None):
    """构建 FastAPI 应用（可注入 service/api_keys，便于测试）。"""
    if not HAS_FASTAPI:
        raise RuntimeError(MISSING_DEPS_HINT)
    from .config import load_config

    cfg = cfg if cfg is not None else load_config()
    server = (cfg.get("server") or {}) if isinstance(cfg, dict) else {}
    keys = resolve_api_keys(cfg) if api_keys is None else list(api_keys)
    limit = server.get("rate_limit_per_min", DEFAULT_RATE_LIMIT) if rate_limit is None else rate_limit
    try:
        limit = int(limit or 0)
    except (TypeError, ValueError):
        limit = DEFAULT_RATE_LIMIT
    svc = service or ForgeService(cfg=cfg, approve_mode=server.get("approve_mode", "auto_reject"),
                                  store=store)
    limiter = RateLimiter(limit)

    app = FastAPI(title="forge", version=__version__, description="forge API（#14 服务化）")
    app.state.service = svc
    app.state.api_keys = keys
    app.state.rate_limiter = limiter

    def auth(request: Request, authorization: str = Header(default=None),
             x_api_key: str = Header(default=None)):
        """鉴权 + 限流依赖：返回调用方标识。"""
        provided = extract_key(authorization, x_api_key)
        host = request.client.host if request.client else ""
        ok, reason = check_auth(keys, provided, host)
        if not ok:
            raise HTTPException(status_code=401, detail=reason,
                                headers={"WWW-Authenticate": "Bearer"})
        ident = identity_of(keys, provided, host)
        wait = limiter.hit(ident)
        if wait:
            raise HTTPException(status_code=429, detail=f"请求过于频繁，请 {wait:.0f}s 后重试",
                                headers={"Retry-After": str(int(wait) + 1)})
        return ident

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t0 = time.time()
        response = await call_next(request)
        ms = (time.time() - t0) * 1000
        sid = ""
        try:
            sid = (request.path_parameters or {}).get("sid") or ""
        except Exception:
            pass
        client = request.client.host if request.client else "-"
        try:
            print(f"[forge] {request.method} {request.url.path} {response.status_code} "
                  f"{ms:.0f}ms client={client}"
                  + (f" session={sid}" if sid else ""), flush=True)
        except Exception:
            pass
        # #7 完整日志：同一条请求也落结构化日志（可被采集/分析，含脱敏）
        try:
            from .logging_setup import log_event

            log_event("INFO" if response.status_code < 500 else "ERROR", "http_request",
                      method=request.method, path=request.url.path,
                      status=response.status_code, ms=ms, client=client, session=sid)
        except Exception:
            pass
        response.headers["X-Forge-Version"] = __version__
        return response

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__}

    @app.get("/api/status")
    def api_status(_ident: str = Depends(auth)):
        return svc.status(keys=keys, rate_limit=limit)

    @app.post("/api/sessions")
    async def create_session(request: Request, _ident: str = Depends(auth)):
        body = await _json_body(request)
        title = body.get("title") if isinstance(body, dict) else None
        sid = svc.create_session(title=title)
        return {"session_id": sid, "session": svc.store.get(sid)}

    @app.get("/api/sessions")
    def list_sessions(limit_n: int = 50, _ident: str = Depends(auth)):
        return {"sessions": svc.store.list_sessions(limit=limit_n)}

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str, _ident: str = Depends(auth)):
        s = svc.store.get(sid)
        if not s:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"session": s, "messages": svc.store.history(sid)}

    @app.delete("/api/sessions/{sid}")
    def delete_session(sid: str, _ident: str = Depends(auth)):
        if not svc.store.delete(sid):
            raise HTTPException(status_code=404, detail="会话不存在")
        svc.forget(sid)
        return {"ok": True, "deleted": sid}

    @app.post("/api/chat")
    async def api_chat(request: Request, _ident: str = Depends(auth)):
        body = await _json_body(request)
        message = str(body.get("message") or "").strip()
        sid = body.get("session_id") or None
        if not message:
            raise HTTPException(status_code=400, detail="message 不能为空")
        if sid and not svc.store.get(sid):
            raise HTTPException(status_code=404, detail="会话不存在")
        try:
            return await svc.chat(message, session_id=sid)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # 模型/工具异常不裸抛 500 堆栈给调用方
            try:
                from .logging_setup import log_event

                log_event("ERROR", "chat_failed", session=sid, error=str(e))
            except Exception:
                pass
            return JSONResponse(status_code=502,
                                content={"error": "处理失败", "detail": str(e), "session_id": sid})


    # ════════════════════════════════════════════════════════════
    # #17 客户端执行器：中心侧接口（客户端**只出站**长轮询，反向连接，A25）
    # ════════════════════════════════════════════════════════════
    from .executor_hub import ExecutorError, get_hub

    hub = hub if hub is not None else get_hub(cfg if isinstance(cfg, dict) else None)
    app.state.executor_hub = hub

    def _hub_http(e):
        """把枢纽的错误码映射成合适的 HTTP 状态码。"""
        if e.code in ("E_UNKNOWN_DEVICE", "E_OFFLINE"):
            return 404
        if e.code in ("E_CAP_DENIED", "E_CAP_UNSUPPORTED", "E_STAGE_READONLY", "E_NEED_APPROVAL"):
            return 403
        if e.code == "E_QUEUE_FULL":
            return 429
        return 400

    @app.post("/api/executor/register")
    async def executor_register(request: Request, _ident: str = Depends(auth)):
        """客户端执行器注册（声明身份与能力）。"""
        body = await _json_body(request)
        try:
            return await run_in_threadpool(hub.register, body)
        except ExecutorError as e:
            raise HTTPException(status_code=_hub_http(e), detail=f"{e.code}: {e.message}")

    @app.post("/api/executor/poll")
    async def executor_poll(request: Request, _ident: str = Depends(auth)):
        """客户端长轮询领命令（挂起等待，同时充当心跳）。"""
        body = await _json_body(request)
        device_id = str(body.get("device_id") or "")
        try:
            cmds = await run_in_threadpool(hub.poll, device_id, body.get("wait"), body.get("max_commands") or 1)
        except ExecutorError as e:
            raise HTTPException(status_code=_hub_http(e), detail=f"{e.code}: {e.message}")
        return {"ok": True, "device_id": device_id, "commands": cmds,
                "server_time": round(time.time(), 3)}

    @app.post("/api/executor/result")
    async def executor_result(request: Request, _ident: str = Depends(auth)):
        """客户端回结果（成功/失败 + 输出 + 可选截图）。"""
        body = await _json_body(request)
        ok = await run_in_threadpool(hub.submit_result, body.get("device_id"), body.get("seq"),
                                     body.get("ok"), body.get("output", ""), body.get("error", ""),
                                     body.get("extra"))
        if not ok:
            raise HTTPException(status_code=404, detail="未知 seq（命令可能已超时）")
        return {"ok": True}

    @app.get("/api/executor/devices")
    def executor_devices(online_only: int = 0, _ident: str = Depends(auth)):
        """在线执行器列表 + 枢纽统计。"""
        return {"devices": hub.devices(only_online=bool(online_only)), "hub": hub.stats()}

    @app.post("/api/executor/command")
    async def executor_command(request: Request, _ident: str = Depends(auth)):
        """派一条命令给执行器并等结果（工具与四角色循环都走这里）。"""
        body = await _json_body(request)
        try:
            cmd = await run_in_threadpool(hub.dispatch, str(body.get("device_id") or ""),
                                          str(body.get("cap") or ""), str(body.get("action") or ""),
                                          body.get("args") or {}, body.get("timeout"),
                                          bool(body.get("approved")))
        except ExecutorError as e:
            raise HTTPException(status_code=_hub_http(e), detail=f"{e.code}: {e.message}")
        return {"ok": cmd.ok, "command": cmd.as_dict(), "extra": cmd.extra}

    @app.post("/api/executor/cua")
    async def executor_cua(request: Request, _ident: str = Depends(auth)):
        """Computer Use 任务（四角色分离：规划/执行/评估/监督，A25）。"""
        body = await _json_body(request)
        from .cua import CuaError, get_cua

        cua = get_cua(cfg if isinstance(cfg, dict) else None, hub=hub)
        try:
            res = await cua.run(str(body.get("task") or ""),      # run 是 async（内部把阻塞的 dispatch 放线程）
                                body.get("device_id"), body.get("max_steps"))
        except (CuaError, ExecutorError) as e:
            raise HTTPException(status_code=_hub_http(e) if isinstance(e, ExecutorError) else 400,
                                detail=str(e))
        return res.as_dict()
    return app


async def _json_body(request):
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════
# 启动入口
# ══════════════════════════════════════════════════════════════════
def resolve_bind(host=None, port=None, cfg=None):
    """解析监听地址：CLI 参数 > 配置 server.host/port > 默认值。"""
    if cfg is None:
        from .config import load_config

        cfg = load_config()
    server = (cfg.get("server") or {}) if isinstance(cfg, dict) else {}
    host = host or server.get("host") or DEFAULT_HOST
    try:
        port = int(port or server.get("port") or DEFAULT_PORT)
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    return host, port


def serve(host=None, port=None, cfg=None, log_level="info", store=None):
    """阻塞启动（forge --serve）。Ctrl-C 停止。"""
    if not HAS_FASTAPI or not HAS_UVICORN:
        raise RuntimeError(MISSING_DEPS_HINT)
    host, port = resolve_bind(host, port, cfg)
    app = create_app(cfg=cfg, store=store)
    keys = app.state.api_keys
    print(f"  🌐 forge API 已启动：http://{host}:{port}/  （文档 /docs）")
    print(f"  🔐 鉴权：{'API key（{} 个）'.format(len(keys)) if keys else '未配 key → 仅本机可访问'}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def start_background(host=None, port=None, cfg=None, timeout=15.0, store=None):
    """后台线程启动（REPL 里 /serve 用）；返回 (server, url)。停止：server.should_exit = True。"""
    if not HAS_FASTAPI or not HAS_UVICORN:
        raise RuntimeError(MISSING_DEPS_HINT)
    host, port = resolve_bind(host, port, cfg)
    app = create_app(cfg=cfg, store=store)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True, name="forge-serve")
    th.start()
    deadline = time.time() + timeout
    while time.time() < deadline and not getattr(server, "started", False):
        if not th.is_alive():
            raise RuntimeError("服务启动失败（端口被占用？换 --port 再试）")
        time.sleep(0.1)
    return server, f"http://{host}:{port}/"
