"""自动任务调度器（#18 自动任务，进程内调度）。

让 forge 在运行期间后台自动执行周期/定时任务：
  - 支持三类调度：interval（每 N 小时/分钟/天）、daily（每天 HH:MM）、once（指定时间一次性）
  - SQLite 持久化（data/tasks.db），重启后保留并补跑离线期间到期的任务
  - 后台线程 + 独立事件循环跑 Agent.run（非流式、写操作自动放行），不阻塞交互式 REPL
  - 每次执行结果写 runs 表；可选 kb_sink 把结果沉淀进知识库
  - 与现有模块正交：复用 Agent / 知识库 / 熔断 / 记忆，不引入新依赖

设计取舍（与 A26 daemon 区分）：本模块是「进程内调度」，forge 开着时后台跑；
不做跨进程常驻 daemon（那是 M5 部署层的事，见 A25/A26）。
"""

import asyncio
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB = os.path.join(_ROOT, "data", "tasks.db")
_TICK = 15  # 调度轮询间隔（秒）


def _now() -> datetime:
    return datetime.now()


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _from_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


# ====================== 调度表达式解析 ======================

class ScheduleError(ValueError):
    pass


def parse_schedule(text: str):
    """解析中/英调度表达式 → (sched_type, expr, human)。

    - interval：每2小时 / 每30分钟 / 每1天 / every 2 hours / every 30 minutes
    - daily：  每天09:00 / 每日9:00 / daily 09:00
    - once：   once 2026-08-20T14:00 / 一次 2026-08-20 14:00
    """
    t = text.strip()
    if not t:
        raise ScheduleError("调度表达式不能为空")

    # —— interval ——
    m = re.search(r"(?:每|every)\s*(\d+)\s*(小时|分钟|天|h|m|hours?|minutes?|days?)", t, re.I)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit in ("小时", "h", "hour", "hours"):
            secs = n * 3600
        elif unit in ("分钟", "m", "minute", "minutes"):
            secs = n * 60
        else:  # 天
            secs = n * 86400
        if secs <= 0:
            raise ScheduleError("间隔必须为正数")
        return ("interval", str(secs), f"每 {n} {unit}")

    # —— daily ——
    m = re.search(r"(?:每?天|每日|每天|daily)\s*(\d{1,2})[:：](\d{2})", t, re.I)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ScheduleError("时间超出范围")
        return ("daily", f"{hh:02d}:{mm:02d}", f"每天 {hh:02d}:{mm:02d}")

    # —— once ——
    m = re.search(r"(?:一次|once|at)\s*(.+)", t, re.I)
    if m:
        raw = m.group(1).strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(raw, fmt)
                return ("once", _iso(dt), f"一次性 {_iso(dt)}")
            except ValueError:
                continue
        raise ScheduleError(f"无法解析一次性时间：{raw}")

    raise ScheduleError(
        "不支持的调度格式。示例：每2小时 / 每30分钟 / 每天09:00 / once 2026-08-20T14:00"
    )


def compute_next(sched_type: str, expr: str, after: datetime) -> datetime:
    """给定调度类型与 expr，返回 after 之后的下一个触发时间。"""
    if sched_type == "interval":
        return after + timedelta(seconds=int(expr))
    if sched_type == "daily":
        hh, mm = map(int, expr.split(":"))
        cand = after.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if cand <= after:
            cand += timedelta(days=1)
        return cand
    if sched_type == "once":
        return _from_iso(expr)
    raise ScheduleError(f"未知调度类型：{sched_type}")


# ====================== 调度器 ======================

class TaskScheduler:
    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None
        self._init_db()

    # ---------- 存储 ----------
    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.execute("PRAGMA journal_mode=MEMORY")  # 沙箱下默认 DELETE 锁会挂起
        return c

    def _init_db(self):
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                sched_type TEXT,
                expr TEXT,
                prompt TEXT,
                role TEXT,
                kb_sink INTEGER DEFAULT 0,
                enabled INTEGER DEFAULT 1,
                created_at TEXT,
                last_run TEXT,
                next_run TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                task_name TEXT,
                started_at TEXT,
                finished_at TEXT,
                ok INTEGER,
                output TEXT,
                error TEXT
            )""")

    # ---------- 增删改查 ----------
    def add(self, name: str, schedule_text: str, prompt: str, role: str = "default",
            kb_sink: bool = False) -> str:
        name = (name or "").strip()
        prompt = (prompt or "").strip()
        if not name:
            return "⚠ 任务名不能为空"
        if not prompt:
            return "⚠ 任务提示词不能为空"
        try:
            stype, expr, human = parse_schedule(schedule_text)
        except ScheduleError as e:
            return f"⚠ 调度解析失败：{e}"
        nxt = compute_next(stype, expr, _now())
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO tasks(name, sched_type, expr, prompt, role, kb_sink, enabled, created_at, next_run)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (name, stype, expr, prompt, role, 1 if kb_sink else 0,
                 _iso(_now()), _iso(nxt)),
            )
        return f"✅ 已登记自动任务「{name}」（{human}），下次执行：{_iso(nxt)}"

    def delete(self, name: str) -> str:
        with self._lock, self._conn() as c:
            r = c.execute("DELETE FROM tasks WHERE name=?", (name,)).rowcount
        return f"✅ 已删除任务「{name}」" if r else f"⚠ 未找到任务「{name}」"

    def set_enabled(self, name: str, enabled: bool) -> str:
        with self._lock, self._conn() as c:
            r = c.execute("UPDATE tasks SET enabled=? WHERE name=?",
                          (1 if enabled else 0, name)).rowcount
        if not r:
            return f"⚠ 未找到任务「{name}」"
        # 重新计算下次执行时间
        if enabled:
            self._reschedule(name)
        return f"✅ 已{'启用' if enabled else '停用'}任务「{name}」"

    def list_tasks(self) -> list:
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT name, sched_type, expr, prompt, role, kb_sink, enabled, created_at, last_run, next_run "
                "FROM tasks ORDER BY id"
            ).fetchall()
        return [
            {
                "name": r[0], "sched_type": r[1], "expr": r[2], "prompt": r[3],
                "role": r[4], "kb_sink": bool(r[5]), "enabled": bool(r[6]),
                "created_at": r[7], "last_run": r[8], "next_run": r[9],
            }
            for r in rows
        ]

    def get(self, name: str):
        for t in self.list_tasks():
            if t["name"] == name:
                return t
        return None

    def recent_runs(self, name: str = None, limit: int = 10) -> list:
        with self._lock, self._conn() as c:
            if name:
                rows = c.execute(
                    "SELECT task_name, started_at, finished_at, ok, output, error FROM runs "
                    "WHERE task_name=? ORDER BY id DESC LIMIT ?", (name, limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT task_name, started_at, finished_at, ok, output, error FROM runs "
                    "ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [
            {"task_name": r[0], "started_at": r[1], "finished_at": r[2],
             "ok": bool(r[3]), "output": r[4], "error": r[5]}
            for r in rows
        ]

    def clear_runs(self, name: str = None) -> int:
        with self._lock, self._conn() as c:
            if name:
                n = c.execute("DELETE FROM runs WHERE task_name=?", (name,)).rowcount
            else:
                n = c.execute("DELETE FROM runs").rowcount
        return n

    # ---------- 执行 ----------
    def _reschedule(self, name: str):
        t = self.get(name)
        if not t:
            return
        if t["sched_type"] == "once":
            with self._lock, self._conn() as c:
                c.execute("UPDATE tasks SET enabled=0, next_run=NULL WHERE name=?", (name,))
            return
        nxt = compute_next(t["sched_type"], t["expr"], _now())
        with self._lock, self._conn() as c:
            c.execute("UPDATE tasks SET next_run=? WHERE name=?", (_iso(nxt), name))

    def _run_async(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _execute(self, task: dict) -> dict:
        """在独立事件循环里跑一次任务，落库并返回 {ok, output, error, tokens}。"""
        from .agent import Agent
        from .approval import Approver
        started = _now()
        ok, output, error, tokens = True, "", "", {}
        try:
            agent = Agent(stream=False, role=task["role"],
                          approver=Approver(auto_approve=True))
            output = self._run_async(agent.run(task["prompt"]))
            tokens = dict(agent.total_tokens)
            # 可选沉淀进知识库
            if task.get("kb_sink") and output:
                try:
                    from .tools import _get_kb
                    title = f"[自动任务] {task['name']} @ {_iso(started)}"
                    _get_kb().add(title, output)
                except Exception as e:
                    error = (error + f"；知识库沉淀失败：{e}").strip("；")
        except Exception as e:
            ok = False
            error = str(e)
            output = ""
        finished = _now()
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO runs(task_name, started_at, finished_at, ok, output, error) VALUES(?,?,?,?,?,?)",
                (task["name"], _iso(started), _iso(finished), 1 if ok else 0, output, error),
            )
            c.execute("UPDATE tasks SET last_run=? WHERE name=?", (_iso(finished), task["name"]))
        return {"ok": ok, "output": output, "error": error, "tokens": tokens,
                "started": started, "finished": finished}

    def run_now(self, name: str) -> str:
        t = self.get(name)
        if not t:
            return f"⚠ 未找到任务「{name}」"
        if not t["enabled"]:
            return f"⚠ 任务「{name}」已停用，先 /task on {name}"
        res = self._execute(t)
        self._reschedule(name)
        if res["ok"]:
            return f"✅ 任务「{name}」执行完成（{_iso(res['finished'])}）\n{res['output'][:500]}"
        return f"⚠ 任务「{name}」执行失败：{res['error']}"

    # ---------- 后台线程 ----------
    def _notify(self, msg: str):
        try:
            print("\n" + "─" * 8 + f" 🔔 自动任务 {msg} " + "─" * 8)
        except Exception:
            pass

    def _catch_up(self):
        """启动补跑：离线期间到期且仍启用的任务，各补跑一次并重新排期。"""
        now = _now()
        due = [t for t in self.list_tasks()
               if t["enabled"] and t["next_run"] and _from_iso(t["next_run"]) <= now]
        for t in due:
            self._notify(f"补跑离线到期任务「{t['name']}」")
            try:
                res = self._execute(t)
                self._reschedule(t["name"])
                self._notify(f"「{t['name']}」{'完成' if res['ok'] else '失败：'+str(res['error'])}")
            except Exception as e:
                self._notify(f"「{t['name']}」补跑异常：{e}")

    def _loop(self):
        try:
            self._catch_up()
        except Exception as e:
            self._notify(f"启动补跑异常：{e}")
        while not self._stop:
            try:
                now = _now()
                due = [t for t in self.list_tasks()
                       if t["enabled"] and t["next_run"] and _from_iso(t["next_run"]) <= now]
                for t in due:
                    try:
                        res = self._execute(t)
                        self._reschedule(t["name"])
                        self._notify(
                            f"「{t['name']}」{'完成' if res['ok'] else '失败：'+str(res['error'])}"
                            + (f"（{res['output'][:80]}…）" if res["ok"] and res["output"] else "")
                        )
                    except Exception as e:
                        self._notify(f"「{t['name']}」执行异常：{e}")
                        try:
                            self._reschedule(t["name"])
                        except Exception:
                            pass
            except Exception as e:
                self._notify(f"调度循环异常：{e}")
            # 分段睡眠，保证 stop 能及时响应
            for _ in range(_TICK):
                if self._stop:
                    break
                time.sleep(1)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="forge-tasks", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)


_scheduler = None


def get_scheduler() -> TaskScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
