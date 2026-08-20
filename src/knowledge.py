"""M3 #15 知识库：SQLite + FTS5 内嵌全文检索（对应模块 #15 / A15 知识库）。

定位：**索引库即源文档**——知识直接沉淀进库内（inline 条目），不依赖外部源文件；
外部文件（ingest/sync）是可选的导入通道。

能力：
  - add(title, content)：直接写入知识条目（标题重复覆盖更新）——对话里 kb_add 工具就是走这里
  - ingest(path)：外部目录/单文件导入（可选）
  - sync(path)：目录同步 + 孤儿清理（只清文件索引，不动库内条目）
  - query(text, limit)：FTS5 全文检索（trigram 中文子串 + LIKE 兜底）
  - list_entries / delete / export_entries：库内条目管理
  - stats()：文档数与总字符

设计：docs 表（path 主键 + kind: file=外部文件索引 / inline=库内源条目）关联
fts5 虚拟表（title + content 全文索引）；trigram 不可用时降级 unicode61。
MCP Server 暴露留 M5 之后。
"""
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager

TEXT_EXTS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".html", ".csv", ".log", ".js", ".ts"}
MAX_FILE_BYTES = 1_000_000  # 单文件 >1MB 跳过（防索引爆炸）

INLINE_PREFIX = "kb://"  # 库内条目的 path 标识

_FTS_CHARS = re.compile(r'["()*\-^]')


def _fts_escape(s: str) -> str:
    """FTS5 查询转义：特殊字符替换为空格，保留中文/字母/数字。"""
    return _FTS_CHARS.sub(" ", s).strip()


class KnowledgeBase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        """连接上下文：MEMORY journal（沙箱文件系统不支持 DELETE journal 锁，会挂起）+ 用完必关。"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=MEMORY")  # 关键：沙箱下默认 DELETE 锁会挂起
            conn.execute("PRAGMA synchronous=OFF")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS docs ("
                      "path TEXT PRIMARY KEY, title TEXT, updated REAL, chars INTEGER, kind TEXT DEFAULT 'file')")
            cols = [r[1] for r in c.execute("PRAGMA table_info(docs)").fetchall()]
            if "kind" not in cols:
                c.execute("ALTER TABLE docs ADD COLUMN kind TEXT DEFAULT 'file'")  # 老库迁移
            c.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            try:
                c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, content, tokenize='trigram')")
            except sqlite3.OperationalError:
                c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, content)")  # 老 SQLite 降级

    # ---------- meta（同步目录登记等） ----------

    def set_meta(self, key, value):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))

    def get_meta(self, key, default=None):
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            if not row:
                return default
            try:
                return json.loads(row[0])
            except (TypeError, ValueError):
                return default

    def sync_dirs(self):
        """已登记的自动同步目录列表。"""
        return self.get_meta("sync_dirs", []) or []

    # ---------- 建索引 ----------

    def ingest(self, path: str):
        """建索引：目录递归或单文件；返回 (新增数, 跳过数)。"""
        files = []
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                for n in names:
                    p = os.path.join(root, n)
                    if os.path.splitext(p)[1].lower() in TEXT_EXTS:
                        files.append(p)
        elif os.path.isfile(path):
            files.append(path)
        added = skipped = 0
        for p in files:
            try:
                if os.path.getsize(p) > MAX_FILE_BYTES:
                    skipped += 1
                    continue
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                mtime = os.path.getmtime(p)
                with self._conn() as c:
                    cur = c.execute("SELECT rowid, updated FROM docs WHERE path=?", (p,)).fetchone()
                    if cur and cur[1] >= mtime:
                        continue  # 已索引且未变化 → 跳过
                    if cur:
                        c.execute("DELETE FROM fts WHERE rowid=?", (cur[0],))
                        c.execute("UPDATE docs SET title=?, updated=?, chars=? WHERE path=?",
                                  (os.path.basename(p), mtime, len(content), p))
                        rid = cur[0]
                    else:
                        r = c.execute("INSERT INTO docs(path,title,updated,chars) VALUES(?,?,?,?)",
                                      (p, os.path.basename(p), mtime, len(content)))
                        rid = r.lastrowid
                    c.execute("INSERT INTO fts(rowid, title, content) VALUES(?,?,?)",
                              (rid, os.path.basename(p), content))
                added += 1
            except OSError:
                skipped += 1  # 文件被占用/无权限等，跳过不崩
        return added, skipped

    # ---------- 同步（自动整理） ----------

    def sync(self, path: str):
        """同步目录/文件到索引，保持与源一致：新增 / 变更重建 / 源已删除的孤儿索引清理。

        只清理「索引路径属于该同步目标」的孤儿（不影响其他目录手动 ingest 的索引）。
        返回 (added, updated, removed)。
        """
        target = os.path.abspath(path)
        files = []
        if os.path.isdir(target):
            for root, _, names in os.walk(target):
                for n in names:
                    p = os.path.join(root, n)
                    if os.path.splitext(p)[1].lower() in TEXT_EXTS:
                        files.append(p)
        elif os.path.isfile(target):
            files.append(target)
        else:
            return 0, 0, 0

        added = updated = removed = 0
        # 1) 新增 / 变更
        for p in files:
            try:
                if os.path.getsize(p) > MAX_FILE_BYTES:
                    continue
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                mtime = os.path.getmtime(p)
                with self._conn() as c:
                    cur = c.execute("SELECT rowid, updated FROM docs WHERE path=?", (p,)).fetchone()
                    if cur:
                        if cur[1] >= mtime:
                            continue  # 未变化
                        c.execute("DELETE FROM fts WHERE rowid=?", (cur[0],))
                        c.execute("UPDATE docs SET title=?, updated=?, chars=? WHERE path=?",
                                  (os.path.basename(p), mtime, len(content), p))
                        c.execute("INSERT INTO fts(rowid, title, content) VALUES(?,?,?)",
                                  (cur[0], os.path.basename(p), content))
                        updated += 1
                    else:
                        r = c.execute("INSERT INTO docs(path,title,updated,chars) VALUES(?,?,?,?)",
                                      (p, os.path.basename(p), mtime, len(content)))
                        c.execute("INSERT INTO fts(rowid, title, content) VALUES(?,?,?)",
                                  (r.lastrowid, os.path.basename(p), content))
                        added += 1
            except OSError:
                continue  # 文件被占用/无权限，跳过不崩
        # 2) 孤儿清理：索引里属于该目标目录、但源文件已不存在
        prefix = target + os.sep
        if os.path.isfile(target):
            prefix = os.path.dirname(target) + os.sep
        with self._conn() as c:
            rows = c.execute("SELECT rowid, path FROM docs WHERE path LIKE ?", (prefix + "%",)).fetchall()
            for rid, p in rows:
                if not os.path.exists(p):
                    c.execute("DELETE FROM fts WHERE rowid=?", (rid,))
                    c.execute("DELETE FROM docs WHERE rowid=?", (rid,))
                    removed += 1
        return added, updated, removed

    def register_sync_dir(self, path: str):
        """登记自动同步目录（启动时 forge 会静默重放）。去重保留。"""
        dirs = self.sync_dirs()
        p = os.path.abspath(path)
        if p not in dirs:
            dirs.append(p)
            self.set_meta("sync_dirs", dirs)
        return dirs

    # ---------- 库内条目（索引库即源文档） ----------

    def add(self, title: str, content: str, kind: str = "inline"):
        """直接写入知识条目（不依赖外部文件）。标题重复时覆盖更新。

        返回 (ok, msg)。kind=inline 是库内源条目；kind=file 仅内部 ingest 用。
        """
        title = (title or "").strip()
        if not title:
            return False, "标题不能为空"
        content = content or ""
        now = time.time()
        with self._conn() as c:
            cur = c.execute("SELECT rowid, path FROM docs WHERE kind=? AND title=?",
                            (kind, title)).fetchone()
            if cur:
                c.execute("DELETE FROM fts WHERE rowid=?", (cur[0],))
                c.execute("UPDATE docs SET updated=?, chars=? WHERE rowid=?", (now, len(content), cur[0]))
                c.execute("INSERT INTO fts(rowid, title, content) VALUES(?,?,?)", (cur[0], title, content))
                return True, f"已更新条目「{title}」"
            path = f"{INLINE_PREFIX}{uuid.uuid4().hex[:12]}"
            r = c.execute("INSERT INTO docs(path,title,updated,chars,kind) VALUES(?,?,?,?,?)",
                          (path, title, now, len(content), kind))
            c.execute("INSERT INTO fts(rowid, title, content) VALUES(?,?,?)", (r.lastrowid, title, content))
            return True, f"已写入条目「{title}」"

    def list_entries(self, kind: str = "inline"):
        """列出库内条目（按更新时间倒序）。返回 [{title, chars, updated, path}]。"""
        with self._conn() as c:
            rows = c.execute("SELECT title, chars, updated, path FROM docs WHERE kind=? ORDER BY updated DESC",
                             (kind,)).fetchall()
            return [{"title": r[0], "chars": r[1], "updated": r[2], "path": r[3]} for r in rows]

    def delete(self, identifier: str):
        """删除条目（按标题或 kb:// path）。返回 (ok, msg)。"""
        with self._conn() as c:
            cur = c.execute("SELECT rowid FROM docs WHERE path=? OR title=?",
                            (identifier, identifier)).fetchone()
            if not cur:
                return False, f"找不到条目「{identifier}」"
            c.execute("DELETE FROM fts WHERE rowid=?", (cur[0],))
            c.execute("DELETE FROM docs WHERE rowid=?", (cur[0],))
            return True, f"已删除条目「{identifier}」"

    def export_entries(self, out_dir: str, identifier: str | None = None):
        """库内条目导出为 Markdown 文件（索引库 → 可读源文档出口）。

        identifier 为空导出全部；否则导出匹配标题/path 的单条。
        返回导出的文件路径列表。
        """
        os.makedirs(out_dir, exist_ok=True)
        entries = self.list_entries()
        if identifier:
            entries = [e for e in entries if e["title"] == identifier or e["path"] == identifier]
        exported = []
        for e in entries:
            with self._conn() as c:
                row = c.execute(
                    "SELECT content FROM fts WHERE rowid=(SELECT rowid FROM docs WHERE path=?)",
                    (e["path"],)).fetchone()
            if not row:
                continue
            safe = re.sub(r'[\\/:*?"<>|]', "_", e["title"]) or "untitled"
            p = os.path.join(out_dir, f"{safe}.md")
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(e["updated"]))
            with open(p, "w", encoding="utf-8") as f:
                f.write(f"# {e['title']}\n\n> 来源：forge 知识库 · {stamp}\n\n{row[0]}\n")
            exported.append(p)
        return exported

    # ---------- 检索 ----------

    def query(self, text: str, limit: int = 5):
        """全文检索；返回 [{title, path, snippet, score}]，无命中返回 []。

        召回优先：查询词按空格拆分、OR 组合（任一命中即可）；FTS 空结果一律 LIKE 兜底
        （覆盖 trigram 对 2 字中文词/整串不连写词的漏检）。
        """
        q = _fts_escape(text or "")
        words = [w for w in q.split() if w]
        if not words:
            return []
        with self._conn() as c:
            rows = []
            match = " OR ".join(f'"{w}"' for w in words)
            try:
                rows = c.execute(
                    "SELECT d.title, d.path, snippet(fts, 1, '⟪', '⟫', '…', 12), bm25(fts) "
                    "FROM fts JOIN docs d ON d.rowid = fts.rowid "
                    "WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?", (match, limit)).fetchall()
            except sqlite3.OperationalError:
                pass  # FTS 语法错误等 → 空结果走兜底
            if not rows:
                like = f"%{q}%"
                rows = c.execute(
                    "SELECT d.title, d.path, substr(f.content, 1, 100), 0 FROM fts f "
                    "JOIN docs d ON d.rowid = f.rowid "
                    "WHERE d.title LIKE ? OR d.path LIKE ? OR f.content LIKE ? LIMIT ?",
                    (like, like, like, limit)).fetchall()
            return [{"title": r[0], "path": r[1], "snippet": r[2], "score": round(r[3], 3)} for r in rows]

    # ---------- 统计 ----------

    def stats(self):
        with self._conn() as c:
            n = c.execute("SELECT count(*) FROM docs").fetchone()[0]
            ch = c.execute("SELECT coalesce(sum(chars), 0) FROM docs").fetchone()[0]
            inline = c.execute("SELECT count(*) FROM docs WHERE kind='inline'").fetchone()[0]
            return {"docs": n, "chars": ch, "inline": inline}
