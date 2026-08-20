"""M3 #15 知识库测试：SQLite + FTS5 内嵌全文检索（模块 #15）。

框架无关（unittest.mock），裸跑 python test_knowledge.py 即可。
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

from src.knowledge import KnowledgeBase
from src.tools import TOOLS, execute, get_tools_schema, is_write


def _mk_files(root):
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "sub"), exist_ok=True)
    with open(os.path.join(root, "a.md"), "w", encoding="utf-8") as f:
        f.write("# 本地部署 AI Agent 的成本构成\n算力、模型权重、推理框架 vLLM、网络运维、人员维护。")
    with open(os.path.join(root, "b.txt"), "w", encoding="utf-8") as f:
        f.write("辩论式多智能体：正方反方裁判，多角色异构模型碰撞。")
    with open(os.path.join(root, "sub", "c.py"), "w", encoding="utf-8") as f:
        f.write("# forge 工具系统\nread_file write_file edit_file 只读分级与审批层。")
    with open(os.path.join(root, "skip.bin"), "w", encoding="utf-8") as f:
        f.write("二进制不该被索引" * 100)


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge_kb_")
        self.db = os.path.join(self.tmp, "kb.db")
        self.docs = os.path.join(self.tmp, "docs")
        _mk_files(self.docs)
        self.kb = KnowledgeBase(self.db)

    def test_ingest_dir_and_stats(self):
        added, skipped = self.kb.ingest(self.docs)
        self.assertEqual(added, 3, "md/txt/py 三个文本文件入索引；.bin 不是文本扩展名不算")
        st = self.kb.stats()
        self.assertEqual(st["docs"], 3)
        self.assertGreater(st["chars"], 0)

    def test_ingest_single_file(self):
        self.kb.ingest(os.path.join(self.docs, "a.md"))
        self.assertEqual(self.kb.stats()["docs"], 1)

    def test_query_chinese_hit(self):
        self.kb.ingest(self.docs)
        hits = self.kb.query("本地部署 AI Agent")
        self.assertTrue(hits, "中文多字词应命中")
        self.assertEqual(hits[0]["title"], "a.md")

    def test_query_no_hit(self):
        self.kb.ingest(self.docs)
        self.assertEqual(self.kb.query("不存在的关键词xyz"), [])

    def test_query_two_char_fallback(self):
        # 2 字中文 trigram 漏检 → LIKE 兜底
        self.kb.ingest(self.docs)
        hits = self.kb.query("算力")
        self.assertTrue(hits, "2 字词应通过 LIKE 兜底命中")
        self.assertIn("算力", hits[0]["snippet"])

    def test_ingest_dedup(self):
        self.kb.ingest(self.docs)
        added2, _ = self.kb.ingest(self.docs)
        self.assertEqual(added2, 0, "未变化文件重复 ingest 应跳过")
        self.assertEqual(self.kb.stats()["docs"], 3)

    def test_ingest_reindex_after_change(self):
        p = os.path.join(self.docs, "a.md")
        self.kb.ingest(p)
        with open(p, "a", encoding="utf-8") as f:
            f.write("\n新增内容：审批层与 trace。")
        added, _ = self.kb.ingest(p)
        self.assertEqual(added, 1, "文件变化后应重新索引")
        self.assertTrue(self.kb.query("审批层"), "更新后的内容应可检索")


class TestKbSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge_kbsync_")
        self.db = os.path.join(self.tmp, "kb.db")
        self.docs = os.path.join(self.tmp, "docs")
        os.makedirs(self.docs, exist_ok=True)
        self.kb = KnowledgeBase(self.db)
        with open(os.path.join(self.docs, "a.md"), "w", encoding="utf-8") as f:
            f.write("同步测试文档甲。")
        with open(os.path.join(self.docs, "b.md"), "w", encoding="utf-8") as f:
            f.write("同步测试文档乙。")

    def test_sync_adds_new(self):
        a, u, r = self.kb.sync(self.docs)
        self.assertEqual((a, u, r), (2, 0, 0), "首次 sync 两个文档入库")
        self.assertEqual(self.kb.stats()["docs"], 2)

    def test_sync_updates_changed_and_removes_orphan(self):
        self.kb.sync(self.docs)
        # 改 a.md + 删 b.md
        with open(os.path.join(self.docs, "a.md"), "w", encoding="utf-8") as f:
            f.write("同步测试文档甲，已更新。")
        os.remove(os.path.join(self.docs, "b.md"))
        a, u, r = self.kb.sync(self.docs)
        self.assertEqual((a, u, r), (0, 1, 1), "a 更新、b 孤儿清理")
        self.assertEqual(self.kb.stats()["docs"], 1)
        # 更新后的内容可检索
        self.assertTrue(self.kb.query("已更新"))
        # 已删除的 b 不再可检索
        self.assertFalse(self.kb.query("文档乙"))

    def test_sync_keeps_other_dirs_indexes(self):
        other = os.path.join(self.tmp, "other")
        os.makedirs(other, exist_ok=True)
        with open(os.path.join(other, "keep.md"), "w", encoding="utf-8") as f:
            f.write("别的目录文档，不应被清理。")
        self.kb.sync(self.docs)
        self.kb.ingest(other)
        # 删掉 other 目录的文件后 sync docs —— 不应误删 other 的索引
        os.remove(os.path.join(other, "keep.md"))
        a, u, r = self.kb.sync(self.docs)
        self.assertEqual((a, u, r), (0, 0, 0), "sync docs 不动 other 的索引")
        self.assertEqual(self.kb.stats()["docs"], 3, "other 索引保留")

    def test_register_and_replay(self):
        dirs = self.kb.register_sync_dir(self.docs)
        self.assertIn(os.path.abspath(self.docs), dirs)
        self.kb.register_sync_dir(self.docs)  # 重复登记去重
        self.assertEqual(len(self.kb.sync_dirs()), 1)
        # 重放：新增文件后按登记目录 sync
        with open(os.path.join(self.docs, "c.md"), "w", encoding="utf-8") as f:
            f.write("同步后新增。")
        for d in self.kb.sync_dirs():
            self.kb.sync(d)
        self.assertEqual(self.kb.stats()["docs"], 3)
        self.assertTrue(self.kb.query("同步后新增"))


class TestKbInline(unittest.TestCase):
    """索引库即源文档：库内条目直接沉淀。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="forge_kbinline_")
        self.kb = KnowledgeBase(os.path.join(self.tmp, "kb.db"))

    def test_add_and_query(self):
        ok, msg = self.kb.add("部署心得", "本地部署用 vLLM + AWQ 量化最省显存。")
        self.assertTrue(ok)
        self.assertEqual(self.kb.stats()["inline"], 1)
        hits = self.kb.query("AWQ 量化")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["title"], "部署心得")

    def test_add_overwrite_same_title(self):
        self.kb.add("部署心得", "第一版")
        ok, msg = self.kb.add("部署心得", "第二版，更完善")
        self.assertTrue(ok)
        self.assertIn("更新", msg)
        entries = self.kb.list_entries()
        self.assertEqual(len(entries), 1, "同标题覆盖不新增")
        hits = self.kb.query("更完善")
        self.assertTrue(hits, "覆盖后的内容可检索")

    def test_list_and_delete(self):
        self.kb.add("甲", "内容甲")
        self.kb.add("乙", "内容乙")
        entries = self.kb.list_entries()
        self.assertEqual(len(entries), 2)
        ok, _ = self.kb.delete("甲")
        self.assertTrue(ok)
        self.assertEqual(len(self.kb.list_entries()), 1)
        self.assertEqual(self.kb.list_entries()[0]["title"], "乙")
        ok2, _ = self.kb.delete("不存在")
        self.assertFalse(ok2)

    def test_sync_keeps_inline(self):
        # sync 目录清理孤儿时绝不动库内条目
        docs = os.path.join(self.tmp, "docs")
        os.makedirs(docs, exist_ok=True)
        with open(os.path.join(docs, "a.md"), "w", encoding="utf-8") as f:
            f.write("外部文件文档。")
        self.kb.add("库内条目", "库内沉淀的内容。")
        self.kb.sync(docs)
        # 删掉外部文件再 sync → 只清文件孤儿，库内条目保留
        os.remove(os.path.join(docs, "a.md"))
        a, u, r = self.kb.sync(docs)
        self.assertEqual(r, 1, "清理的是文件孤儿")
        self.assertEqual(self.kb.stats()["inline"], 1, "库内条目不受影响")
        self.assertTrue(self.kb.query("库内沉淀"))

    def test_export_entries(self):
        self.kb.add("导出条目", "导出测试正文。")
        out = os.path.join(self.tmp, "out")
        files = self.kb.export_entries(out)
        self.assertEqual(len(files), 1)
        text = open(files[0], encoding="utf-8").read()
        self.assertIn("# 导出条目", text)
        self.assertIn("导出测试正文", text)
        # 单条导出
        one = self.kb.export_entries(out, "导出条目")
        self.assertEqual(len(one), 1)


class TestKbTools(unittest.TestCase):
    def test_tools_registered(self):
        names = {s["function"]["name"] for s in get_tools_schema()}
        self.assertIn("kb_search", names)
        self.assertIn("kb_ingest", names)
        self.assertTrue(is_write("kb_ingest"), "建索引是写操作，走审批层")
        self.assertFalse(is_write("kb_search"), "检索是只读")

    def test_execute_search(self):
        tmp = tempfile.mkdtemp(prefix="forge_kbtool_")
        p = os.path.join(tmp, "x.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("知识库检索测试：本地部署 AI Agent 应该重视可控安全。")
        # 用临时库替换单例（避免污染真实 data/knowledge.db）
        from src import tools as tools_mod
        from src.knowledge import KnowledgeBase
        real = tools_mod._KB
        tools_mod._KB = KnowledgeBase(os.path.join(tmp, "kb.db"))
        try:
            tools_mod._KB.ingest(p)
            r = json.loads(execute("kb_search", {"query": "可控安全"})["data"])
            self.assertIsInstance(r, list)
            self.assertTrue(r)
            self.assertIn("可控安全", r[0]["snippet"])
        finally:
            tools_mod._KB = real  # 恢复单例

    def test_execute_ingest(self):
        tmp = tempfile.mkdtemp(prefix="forge_kbtool2_")
        p = os.path.join(tmp, "y.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("入库测试文档。")
        from src import tools as tools_mod
        from src.knowledge import KnowledgeBase
        real = tools_mod._KB
        tools_mod._KB = KnowledgeBase(os.path.join(tmp, "kb2.db"))
        try:
            r = json.loads(execute("kb_ingest", {"path": p})["data"])
            self.assertTrue(r["ok"])
            self.assertEqual(r["added"], 1)
            self.assertEqual(r["stats"]["docs"], 1)
        finally:
            tools_mod._KB = real


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if r.wasSuccessful() else 1)
