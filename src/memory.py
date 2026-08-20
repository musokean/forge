"""长期对话记忆：跨会话记住用户偏好 / 事实（区别于 knowledge.py 的文档知识库）。

定位区分：
- knowledge.py = 文档知识库：显式写入的事实/文档（用户说「记到知识库」才存），可导出为源文档
- memory.py     = 用户画像记忆：关于「用户是谁、喜欢什么、习惯怎么干活」的轻量条目，
                  自动/显式沉淀，每次会话启动注入，让 forge 跨会话记得老大

存储：独立 SQLite（data/memory.db，路径可配 memory.db_path），关键词匹配召回（不引入 embedding 依赖）。
"""

from __future__ import annotations
import os
import re
import sqlite3
from contextlib import contextmanager

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PATH = os.path.join(_BASE_DIR, "data", "memory.db")

# 显式记住的用户表达模式（消息命中即自动沉淀，如「记住 / 我喜欢 / 我是 / 我习惯 / 我常 / 我需要 / 我不喜欢」）
_AUTO_PATTERNS = re.compile(
    r"^(记住|请记住|记得|我喜欢|我不喜欢|我讨厌|我是|我是做|我习惯|我常|我一般|我需要|我要|我每天|我负责|我在)"
)


class MemoryStore:
    """长期记忆存储（SQLite + 关键词召回）。"""

    def __init__(self, path: str | None = None):
        self.path = path or _DEFAULT_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        # 沙箱坑：默认 DELETE journal 锁会挂起 → MEMORY journal + 关闭同步（实测秒过）
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS memos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    last_used TEXT
                )"""
            )

    # ---------- 写入 ----------

    def remember(self, content: str) -> tuple[bool, str]:
        """写入一条记忆；内容前 30 字已存在则视为重复（跳过），避免重复沉淀。"""
        content = content.strip()
        if not content:
            return False, "记忆内容为空"
        head = content[:30]
        with self._conn() as c:
            row = c.execute(
                "SELECT id FROM memos WHERE substr(content, 1, 30) = ?", (head,)
            ).fetchone()
            if row:
                return False, "已有相同记忆（自动去重，未重复写入）"
            c.execute("INSERT INTO memos (content) VALUES (?)", (content,))
        return True, "已记住"

    def auto_remember(self, text: str) -> str | None:
        """自动沉淀：消息命中「我喜欢/我是/我习惯…」等模式时，截取整句存入。命中返回内容，否则 None。"""
        text = text.strip()
        m = _AUTO_PATTERNS.match(text)
        if not m:
            return None
        # 取该句（到第一个句号/问号/叹号为止；无标点则取整行前 80 字）
        seg = re.split(r"[。！？!?；;]", text, maxsplit=1)[0].strip()
        if len(seg) > 80:
            seg = seg[:80]
        ok, msg = self.remember(seg)
        return seg if ok else None

    # ---------- 召回 ----------

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """按关键词召回：query 按空格/常见标点切词，OR 匹配 content（小库足够，不引 embedding）。"""
        words = [w for w in re.split(r"[\s,，。.！？!?、;；:：]+", query) if len(w) >= 1]
        if not words:
            return []
        cond = " OR ".join(["content LIKE ?"] * len(words))
        args = [f"%{w}%" for w in words[:8]]
        with self._conn() as c:
            rows = c.execute(
                f"SELECT id, content, created, hit_count FROM memos WHERE {cond} "
                "ORDER BY hit_count DESC, id DESC LIMIT ?",
                (*args, limit),
            ).fetchall()
            ids = [r[0] for r in rows]
            if ids:
                marks = ",".join("?" * len(ids))
                c.execute(f"UPDATE memos SET hit_count = hit_count + 1, last_used = datetime('now','localtime') "
                          f"WHERE id IN ({marks})", ids)
                # 加分后重查，返回更新后的 hit_count（语义一致）
                rows = c.execute(
                    f"SELECT id, content, created, hit_count FROM memos WHERE id IN ({marks})", ids
                ).fetchall()
        return [{"id": r[0], "content": r[1], "created": r[2], "hit_count": r[3]} for r in rows]

    def compose_context(self, query: str, limit: int = 3) -> str:
        """拼一段可注入对话的记忆上下文；无命中返回空串。"""
        mems = self.recall(query, limit)
        if not mems:
            return ""
        lines = "\n".join(f"- {m['content']}" for m in mems)
        return f"[关于你的长期记忆（跨会话，供参考）]\n{lines}"

    # ---------- 管理 ----------

    def forget(self, keyword: str) -> int:
        """删除内容包含 keyword 的记忆，返回删除条数。"""
        with self._conn() as c:
            cur = c.execute("DELETE FROM memos WHERE content LIKE ?", (f"%{keyword}%",))
            return cur.rowcount

    def clear(self) -> int:
        with self._conn() as c:
            cur = c.execute("DELETE FROM memos")
            return cur.rowcount

    def list_all(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, content, created, hit_count FROM memos ORDER BY id DESC").fetchall()
        return [{"id": r[0], "content": r[1], "created": r[2], "hit_count": r[3]} for r in rows]

    def stats(self) -> dict:
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM memos").fetchone()[0]
        return {"count": n, "path": self.path}


_memory: "MemoryStore | None" = None


def get_memory() -> MemoryStore:
    """全局单例（懒加载）。"""
    global _memory
    if _memory is None:
        from .config import load_config
        cfg = load_config()
        p = (cfg.get("memory") or {}).get("db_path")
        _memory = MemoryStore(os.path.join(_BASE_DIR, p) if p and not os.path.isabs(p) else p)
    return _memory


def reset_memory():
    global _memory
    _memory = None
