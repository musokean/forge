"""人脸识别（#18 Phase 2）测试 —— 离线、确定性、不需要摄像头与模型。

覆盖四层：

1. **相似度与裁剪**（`cosine` / `crop_face`）：归一化、正交、零向量、边界裁切、非法框
2. **身份库**（`FaceStore`）：登记/追加/列表/删除/重开持久化 —— 全部走临时 SQLite
3. **匹配判定**（`FaceStore.match`）：**阈值 + 最佳/次佳差距**两条闸门（宁可回「未知」也不硬猜）、
   空库、维度不一致
4. **提取器与工具**（`make_embedder` / `sface_available` / 工具注册）：**没模型就明确报不可用**
   （不假装有）、只读/写分级

用 `StubEmbedder`（按图像内容决定性地出向量）跑通全链路 —— 它不需要 SFace 模型，所以 CI 里真跑。
**真实人脸识别精度不在本文件覆盖范围**（需要模型 + 真人 + 摄像头），`docs/faces.md` 里说明怎么验。

2026-10-03 的取舍：**故意不测 `face_who` / `face_enroll` 的本体**（它们要开摄像头，CI 无摄像头会 flaky）；
这两个工具的门面逻辑（该不该挂、只读还是写）在这里测，取帧/登记链路在 `test_camera.py` 里各自钉住。
"""
import importlib.util
import os
import sys
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:   # pragma: no cover
    np = None

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None
needs_numpy = unittest.skipUnless(HAS_NUMPY, "需要 numpy（pip install \"handcraft-agent[vision]\"）")

sys.path.insert(0, ".")

from forge.faces import (  # noqa: E402
    DEFAULT_MARGIN, DEFAULT_THRESHOLD, FaceError, FaceStore, StubEmbedder,
    available, cosine, crop_face, face_defaults, make_embedder, sface_available,
)

from forge.camera import Face  # noqa: E402


def vec_of(seed_text: str, dim: int = 128):
    """造一个确定性的单位向量（拿字符串当「图像内容」喂给替身提取器）。"""
    class _Img:
        def tobytes(self):
            return seed_text.encode("utf-8")
    return StubEmbedder(dim).embed(_Img())


class TestSimilarity(unittest.TestCase):
    def test_cosine_identical_is_one(self):
        v = vec_of("a")
        self.assertAlmostEqual(cosine(v, v), 1.0, places=6)

    def test_cosine_far_pair_is_small(self):
        a, b = vec_of("alice"), vec_of("bob")
        self.assertLess(cosine(a, b), 0.9)

    @needs_numpy
    def test_cosine_zero_vector_is_zero_not_nan(self):
        self.assertEqual(cosine([0.0] * 128, vec_of("x")), 0.0)

    def test_stub_embedder_is_deterministic_and_normalised(self):
        v1, v2 = vec_of("same"), vec_of("same")
        self.assertEqual(v1, v2)
        self.assertAlmostEqual(sum(x * x for x in v1) ** 0.5, 1.0, places=6)

    def test_stub_embedder_differs_for_different_input(self):
        self.assertNotEqual(vec_of("alice"), vec_of("bob"))

    @needs_numpy
    def test_crop_face_clips_to_image_bounds(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = 255
        out = crop_face(img, Face(38, 38, 24, 24), margin=0.5)     # 外扩后会超出边界
        self.assertEqual(out.shape[0] * out.shape[1] > 0, True)
        self.assertLessEqual(out.shape[1], 100)
        self.assertGreater(int(out.sum()), 0)

    @needs_numpy
    def test_crop_face_rejects_empty_box(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        with self.assertRaises(FaceError):
            crop_face(img, Face(999, 999, 10, 10))


class TestFaceStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="faces-")
        self.db = os.path.join(self.tmp, "faces.db")
        self.store = FaceStore(self.db)
        if not HAS_NUMPY:
            self.skipTest("需要 numpy")

    def tearDown(self):
        self.store.close()

    def test_enroll_new_person_then_append(self):
        r1 = self.store.enroll("alice", [vec_of("alice-1"), vec_of("alice-2")])
        self.assertTrue(r1["new_person"])
        self.assertEqual((r1["added"], r1["total"]), (2, 2))
        r2 = self.store.enroll("alice", [vec_of("alice-3")])
        self.assertFalse(r2["new_person"])
        self.assertEqual((r2["added"], r2["total"]), (1, 3))

    def test_enroll_requires_name_and_vectors(self):
        with self.assertRaises(FaceError):
            self.store.enroll("  ", [vec_of("x")])
        with self.assertRaises(FaceError):
            self.store.enroll("bob", [])

    def test_people_listing_and_counts(self):
        self.store.enroll("alice", [vec_of("a1"), vec_of("a2")])
        self.store.enroll("bob", [vec_of("b1")])
        names = [p["name"] for p in self.store.people()]
        self.assertEqual(names, ["alice", "bob"])
        self.assertEqual(self.store.count_faces(), 3)
        self.assertEqual(self.store.count_faces("alice"), 2)

    def test_forget_removes_all_vectors(self):
        self.store.enroll("carol", [vec_of("c1"), vec_of("c2")])
        res = self.store.forget("carol")
        self.assertEqual((res["found"], res["removed"]), (True, 2))
        self.assertEqual(self.store.count_faces("carol"), 0)
        self.assertEqual([p["name"] for p in self.store.people()], [])
        self.assertFalse(self.store.forget("carol")["found"])

    def test_persists_across_reopen(self):
        self.store.enroll("dave", [vec_of("d1"), vec_of("d2")])
        self.store.close()
        again = FaceStore(self.db)
        try:
            self.assertEqual(again.count_faces("dave"), 2)
            m = again.match(vec_of("d1"))
            self.assertEqual(m.name, "dave")
        finally:
            again.close()

    @needs_numpy
    def test_store_keeps_vectors_not_images(self):
        """隐私硬约束：库里只有向量 BLOB，没有任何图像字段。"""
        self.store.enroll("erin", [vec_of("e1")])
        cols = [r[1] for r in self.store._conn.execute("PRAGMA table_info(faces)").fetchall()]
        self.assertIn("vec", cols)
        for bad in ("image", "img", "photo", "blob_img", "thumbnail"):
            self.assertNotIn(bad, cols)
        dim, blob = self.store._conn.execute("SELECT dim, vec FROM faces LIMIT 1").fetchone()
        self.assertEqual(dim, 128)
        self.assertEqual(len(blob), 128 * 4)             # float32 × 维度


class TestMatch(unittest.TestCase):
    def setUp(self):
        if not HAS_NUMPY:
            self.skipTest("需要 numpy")
        self.tmp = tempfile.mkdtemp(prefix="faces-")
        self.store = FaceStore(os.path.join(self.tmp, "faces.db"))

    def tearDown(self):
        self.store.close()

    def test_exact_vector_matches(self):
        self.store.enroll("alice", [vec_of("alice-1")])
        m = self.store.match(vec_of("alice-1"))
        self.assertFalse(m.unknown)
        self.assertEqual(m.name, "alice")
        self.assertGreater(m.score, 0.99)

    def test_unrelated_faces_are_unknown(self):
        """不相干的人**一批**都不能过阈值（单看一个样本会偶发通过 —— 这正是阈值要看维度的原因）。"""
        self.store.enroll("alice", [vec_of("alice-1")])
        scores = [self.store.match(vec_of(f"stranger-{i}")).score for i in range(10)]
        self.assertLess(max(scores), DEFAULT_THRESHOLD,
                        f"有不相干的人过了阈值 {DEFAULT_THRESHOLD}：最高 {max(scores):.3f}")
        m = self.store.match(vec_of("stranger-0"))
        self.assertTrue(m.unknown)
        self.assertEqual(m.name, "")
        self.assertIn("阈值", m.reason)

    def test_close_pair_is_unknown_by_margin(self):
        """两个人长得像：最佳过阈值但优势不明显 → 必须回「未知」（宁可不认，别认错）。"""
        base = vec_of("twin-base")
        near = [x + 1e-6 for x in base]                  # 与 base 几乎同一个方向
        self.store.enroll("twin-a", [base])
        self.store.enroll("twin-b", [near])
        m = self.store.match(base, threshold=0.3, margin=0.03)
        self.assertTrue(m.unknown, f"不该给名字，却给了 {m.name}（{m.score:.4f} vs {m.runner_up:.4f}）")
        self.assertIn("都像", m.reason)

    def test_empty_store_is_unknown_with_reason(self):
        m = self.store.match(vec_of("x"))
        self.assertTrue(m.unknown)
        self.assertIn("空", m.reason)

    def test_dimension_mismatch_raises(self):
        self.store.enroll("alice", [vec_of("alice-1")])
        with self.assertRaises(FaceError):
            self.store.match([0.1] * 16)

    def test_stats_shape(self):
        self.store.enroll("alice", [vec_of("a1"), vec_of("a2")])
        st = self.store.stats()
        self.assertEqual((st["people"], st["vectors"]), (1, 2))
        self.assertTrue(st["db"].endswith(".db"))


class TestEmbedderFactoryAndTools(unittest.TestCase):
    def test_stub_embedder_factory(self):
        emb = make_embedder("stub")
        self.assertEqual(emb.name, "stub")
        self.assertEqual(len(emb.embed(vec_of("x"))), emb.dim)

    def test_sface_without_model_refuses_cleanly(self):
        """没放模型就给「去哪拿、放哪里」，不假装可用（也不偷偷下载）。"""
        with self.assertRaises(FaceError) as ctx:
            make_embedder("sface", model_path=os.path.join(tempfile.mkdtemp(), "nope.onnx"))
        msg = str(ctx.exception)
        self.assertIn("SFace", msg)
        self.assertIn("opencv_zoo", msg)

    def test_auto_without_model_refuses_with_hint(self):
        if sface_available():
            self.skipTest("本机已装 SFace 模型，auto 会成功")
        with self.assertRaises(FaceError) as ctx:
            make_embedder("auto")
        self.assertIn("没有可用的特征提取器", str(ctx.exception))

    def test_none_embedder_refuses(self):
        with self.assertRaises(FaceError):
            make_embedder("none")

    def test_available_reports_truthfully(self):
        av = available()
        self.assertEqual(av["sface"], sface_available())
        self.assertIn("threshold", av)
        self.assertIn("db", av)

    def test_defaults_are_sane(self):
        d = face_defaults()
        self.assertEqual(d["threshold"], DEFAULT_THRESHOLD)
        self.assertEqual(d["margin"], DEFAULT_MARGIN)
        self.assertGreaterEqual(d["samples"], 1)
        self.assertTrue(str(d["db"]).endswith(".db"))

    @unittest.skipUnless(HAS_NUMPY and HAS_CV2, "工具注册需要 numpy + opencv")
    def test_face_tools_registered_with_readonly_split(self):
        from forge.tools import TOOLS, is_write
        for name in ("face_people", "face_who", "face_enroll", "face_forget"):
            self.assertIn(name, TOOLS, f"{name} 没注册上")
        self.assertFalse(is_write("face_people"), "face_people 只读")
        self.assertFalse(is_write("face_who"), "face_who 只读")
        self.assertTrue(is_write("face_enroll"), "face_enroll 写操作（会改库）")
        self.assertTrue(is_write("face_forget"), "face_forget 写操作（会删数据）")

    def test_face_people_message_when_empty(self):
        """在空库上问名单 → 如实说空，不编造。"""
        if not (HAS_NUMPY and HAS_CV2):
            self.skipTest("需要 numpy + opencv")
        import forge.faces as faces_mod
        tmp = os.path.join(tempfile.mkdtemp(prefix="faces-"), "empty.db")
        orig = faces_mod.face_defaults
        faces_mod.face_defaults = lambda: dict(orig(), db=tmp)
        try:
            from forge.tools import execute
            out = execute("face_people", {})
            self.assertIn("空", str(out))
        finally:
            faces_mod.face_defaults = orig


if __name__ == "__main__":
    unittest.main()
