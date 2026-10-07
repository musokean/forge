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
import shutil
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
    available, cosine, crop_face, embed_face, face_defaults, make_embedder, sface_available,
)

from forge.camera import Face, StubFaceDetector  # noqa: E402


def vec_of(seed_text: str, dim: int = 128):
    """造一个确定性的单位向量（拿字符串当「图像内容」喂给替身提取器）。"""
    class _Img:
        def tobytes(self):
            return seed_text.encode("utf-8")
    return StubEmbedder(dim).embed(_Img())


def _small_image(h=48, w=64, v=120):
    """合成小图（够放下一张小脸的框）—— 全程不碰摄像头 ✓。"""
    return np.full((h, w, 3), v, dtype=np.uint8)


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
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)   # 别把临时库留一地 ✗✓
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


class _LandmarkDetector:
    """替身：像 YuNet 一样能出关键点（但不加载任何模型）。"""

    name = "fake-landmarks"

    def __init__(self, box):
        self.box = box

    def detect(self, image):
        return [self.box]

    def detect_with_landmarks(self, image):
        return [(self.box, "ROW")]


class _AlignEmbedder(StubEmbedder):
    """替身：支持对齐，并记下是否真的收到了关键点行。"""

    def __init__(self):
        super().__init__()
        self.aligned_with = None

    def embed_aligned(self, image, row):
        self.aligned_with = row
        return self.embed(image)


class TestEmbedFace(unittest.TestCase):
    """**有对齐就用对齐**（更准），没有就退回裁剪 —— 三条分支都要覆盖 ✓。"""

    @needs_numpy
    def test_uses_alignment_when_detector_has_landmarks(self):
        emb, img = _AlignEmbedder(), _small_image()
        vec, how = embed_face(emb, img, _LandmarkDetector(Face(4, 4, 8, 8)))
        self.assertEqual(how, "对齐")
        self.assertEqual(emb.aligned_with, "ROW", "关键点行必须传进 alignCrop 那条路")
        self.assertEqual(len(vec), emb.dim)

    @needs_numpy
    def test_falls_back_to_crop_for_haar_like_detector(self):
        _vec, how = embed_face(StubEmbedder(), _small_image(), StubFaceDetector([Face(4, 4, 8, 8)]))
        self.assertEqual(how, "裁剪")

    @needs_numpy
    def test_falls_back_when_embedder_cannot_align(self):
        """检测器有关键点、但提取器不支持对齐（例如替身）→ 安静退回裁剪，不报错 ✓。"""
        _vec, how = embed_face(StubEmbedder(), _small_image(), _LandmarkDetector(Face(4, 4, 8, 8)))
        self.assertEqual(how, "裁剪")

    @needs_numpy
    def test_no_face_raises_a_clear_error(self):
        with self.assertRaises(FaceError):
            embed_face(StubEmbedder(), _small_image(), StubFaceDetector([]))


class TestMatch(unittest.TestCase):
    def setUp(self):
        if not HAS_NUMPY:
            self.skipTest("需要 numpy")
        self.tmp = tempfile.mkdtemp(prefix="faces-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)   # 别把临时库留一地 ✗✓
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

    def test_one_person_many_samples_is_still_recognised(self):
        """**一个人登记多张样本时，也必须认得出他** ✗✓。

        真机实测踩到的 bug：次佳若取「同一人的另一张样本」，margin 永远不过 → 谁都认不出来
        （命中率 2/10）。比较单元必须是「人」而不是「样本」。
        """
        self.store.enroll("solo", [vec_of("s1"), vec_of("s2"), vec_of("s3")])
        m = self.store.match(vec_of("s2"))
        self.assertFalse(m.unknown, f"同一个人的样本之间不该触发 margin：{m.reason}")
        self.assertEqual(m.name, "solo")
        self.assertEqual(m.runner_up, 0.0, "库里只有一个人时，次佳应为 0（没有别人）")

    def test_config_paths_expand_user_home(self):
        """配置里的 `~` 必须展开 ✓ —— 不展开时 sqlite 会去建一个名字就叫 `~` 的目录 ✗。

        用 mock 打配置，**不 reload 模块** ✗（reload 会污染全局状态 → 连坐别的用例 ✗）。
        """
        from unittest.mock import patch
        with patch("forge.config.load_config",
                   return_value={"face": {"db": "~/.forge/faces.db", "samples": 4}}):
            d = face_defaults()
        self.assertTrue(d["db"].startswith(os.path.expanduser("~")), d["db"])
        self.assertNotIn("~", d["db"])
        self.assertEqual(d["samples"], 4, "配置里的值要生效")

    def test_empty_store_is_unknown_with_reason(self):
        m = self.store.match(vec_of("x"))
        self.assertTrue(m.unknown)
        self.assertIn("空", m.reason)

    def test_dimension_mismatch_raises(self):
        self.store.enroll("alice", [vec_of("alice-1")])
        with self.assertRaises(FaceError):
            self.store.match([0.1] * 16)

    def test_mixed_dimension_store_refuses_to_compare(self):
        """换过提取器 → 库里维度混杂：必须明确报错，不能零填充后瞎比（那会认错人）。"""
        self.store.enroll("alice", [vec_of("alice-1")])          # 128 维
        self.store._conn.execute(
            "INSERT INTO faces(person_id, dim, vec, created) VALUES "
            "((SELECT id FROM people WHERE name = 'alice'), 8, ?, 'now')",
            (np.zeros(8, dtype="float32").tobytes(),))
        self.store._conn.commit()
        with self.assertRaises(FaceError) as ctx:
            self.store.match(vec_of("alice-1"))
        self.assertIn("维度", str(ctx.exception))

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
            make_embedder("sface", model_path=os.path.join(tempfile.gettempdir(), "forge-nope-model.onnx"))
        msg = str(ctx.exception)
        self.assertIn("SFace", msg)
        self.assertIn("opencv_zoo", msg)

    def test_auto_without_model_refuses_with_hint(self):
        """`auto` 在**找不到模型**时必须明确报不可用（带去哪儿拿的提示）。

        不依赖本机环境：显式把模型路径指到一个不存在的文件（本机装了真实模型时，
        以前这里会 skip —— 测试随环境跳过是味道 ✗，改成环境无关 ✓）。
        """
        old_env = os.environ.get("FORGE_FACE_MODEL")
        os.environ["FORGE_FACE_MODEL"] = os.path.join(tempfile.gettempdir(), "forge-missing-model.onnx")
        try:
            with self.assertRaises(FaceError) as ctx:
                make_embedder("auto")
            self.assertIn("没有可用的特征提取器", str(ctx.exception))
        finally:
            if old_env is None:
                os.environ.pop("FORGE_FACE_MODEL", None)
            else:
                os.environ["FORGE_FACE_MODEL"] = old_env

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
        _d = tempfile.TemporaryDirectory(prefix="forge-test-")
        self.addCleanup(_d.cleanup)
        tmp = os.path.join(_d.name, "empty.db")
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


class TestNotes(unittest.TestCase):
    """#18 按人记忆最小版：备注写/读/随人删除 ✓。"""

    def test_note_roundtrip_and_people_lists_it(self):
        d = tempfile.TemporaryDirectory(prefix="forge-test-")   # 自动清理 ✓（别用裸 mkdtemp 留一地垃圾 ✗）
        self.addCleanup(d.cleanup)
        tmp = os.path.join(d.name, "n.db")
        st = FaceStore(tmp)
        self.addCleanup(st.close)
        try:
            st.enroll("满仓", [np.ones(128, dtype=np.float32)])
            st.set_note("满仓", "喜欢冰美式")
            self.assertEqual(st.get_note("满仓"), "喜欢冰美式")
            ppl = st.people()
            self.assertEqual(len(ppl), 1)
            self.assertEqual(ppl[0]["note"], "喜欢冰美式")
            self.assertEqual(ppl[0]["faces"], 1)
        finally:
            st.close()

    def test_note_on_unknown_name_raises(self):
        d = tempfile.TemporaryDirectory(prefix="forge-test-")
        self.addCleanup(d.cleanup)
        tmp = os.path.join(d.name, "n2.db")
        st = FaceStore(tmp)
        self.addCleanup(st.close)
        try:
            with self.assertRaises(FaceError):
                st.set_note("查无此人", "x")
            with self.assertRaises(FaceError):
                st.get_note("查无此人")
        finally:
            st.close()

    def test_forget_removes_note_with_person(self):
        d = tempfile.TemporaryDirectory(prefix="forge-test-")
        self.addCleanup(d.cleanup)
        tmp = os.path.join(d.name, "n3.db")
        st = FaceStore(tmp)
        self.addCleanup(st.close)
        try:
            st.enroll("满仓", [np.ones(128, dtype=np.float32)])
            st.set_note("满仓", "临时备注")
            st.forget("满仓")
            self.assertEqual(st.people(), [])
            with self.assertRaises(FaceError):
                st.get_note("满仓")
        finally:
            st.close()

    def test_face_note_tool_is_registered_and_writes(self):
        from forge.tools import TOOLS
        self.assertIn("face_note", TOOLS)
        self.assertFalse(TOOLS["face_note"]["read_only"], "写操作应标 read_only=False ✓")

class TestFaceStoreIsThreadSafe(unittest.TestCase):
    """★ 库必须能被**另一个线程**使用 ✓✓。

    2026-10-06 真机踩到 ✗：探针（现场身份）在后台线程里读库，而 sqlite 连接默认
    `check_same_thread=True` → 真语音轮里识别**永远失败**，报
    「SQLite objects created in a thread can only be used in that same thread」✗。
    而当时的隔离测试是**同线程**调用 → 测不出来 ✗✓。这条测试**故意换一个线程** ✓。
    """

    def setUp(self):
        if not HAS_NUMPY:
            self.skipTest("需要 numpy")
        self.tmp = tempfile.mkdtemp(prefix="faces-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_match_from_another_thread(self):
        import threading
        st = FaceStore(os.path.join(self.tmp, "faces.db"))
        self.addCleanup(st.close)
        vec = [0.05] * 128
        st.enroll("满仓", [vec])
        out = {}

        def worker():
            try:
                m = st.match(vec)
                out["name"] = m.name if m else None
            except Exception as exc:                     # 就是这一句当初会炸 ✗
                out["err"] = "%s: %s" % (type(exc).__name__, exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)
        self.assertNotIn("err", out, "跨线程读库必须可用 ✓（探针就在后台线程里读 ✗）")
        self.assertEqual(out.get("name"), "满仓")

    def test_concurrent_read_while_writing(self):
        """一边读一边写也别炸 ✓（探针读 + 工具写是常态 ✓）。"""
        import threading
        st = FaceStore(os.path.join(self.tmp, "faces2.db"))
        self.addCleanup(st.close)
        st.enroll("满仓", [[0.05] * 128])
        errs = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    st.match([0.05] * 128)
                except Exception as exc:
                    errs.append("%s: %s" % (type(exc).__name__, exc))
                    return

        th = threading.Thread(target=reader)
        th.start()
        for i in range(30):
            st.enroll("翠花", [[0.02] * 128], source="并发写入第%d次" % i)
        stop.set()
        th.join(timeout=10)
        self.assertEqual(errs, [], "并发读写不该出错 ✓")

class TestFaceToolsHoldTheCamera(unittest.TestCase):
    """★★ 脸部工具**必须独占相机** ✓✓ —— 真因就是"两个工具同时开摄像头 = 原生崩" ✗✗。

    2026-10-07 三次同点崩溃（探针+工具 / 探针+突发 / **工具×工具**）；最后一次的日志直接钉死了：
    模型一次并发调用了 `face_people, face_who` ✓，而 `face_who` **不走 `grab_frame`** ✗ ——
    它自己 `OpenCvFrameSource.open()` ✓ → **绕过了 `_CAMERA_LOCK`** ✗ → 两个 `VideoCapture` 同时开
    → OpenCV 原生层终止、**rc=127、无 traceback** ✗✓。

    这条测试把相机换成假的、数**同时在"开"的人数** ✓：必须恒为 1 ✓；并**反证**（撤掉独占）不许通过 ✗。
    """

    class _FakeSource:
        """数并发：进入 open 时 +1，close 时 -1，记录峰值 ✓"""
        peak = 0
        live = 0
        lock = None

        def __init__(self, index=0):
            pass

        def open(self):
            with TestFaceToolsHoldTheCamera._FakeSource.lock:
                TestFaceToolsHoldTheCamera._FakeSource.live += 1
                TestFaceToolsHoldTheCamera._FakeSource.peak = max(
                    TestFaceToolsHoldTheCamera._FakeSource.peak,
                    TestFaceToolsHoldTheCamera._FakeSource.live)
            # ★ 必须**停在 open 里一会儿** ✓✓：否则两次调用根本不重叠 → 测试是空的 ✗✓
            #   （2026-10-07 实测：不加这句，把独占换成空上下文**峰值仍是 1** → 反证反不掉 ✗✗）
            time.sleep(0.15)

        def close(self):
            with TestFaceToolsHoldTheCamera._FakeSource.lock:
                TestFaceToolsHoldTheCamera._FakeSource.live -= 1

        def read(self):
            return None

    class _FakeDetector:
        def detect(self, image):
            return []

    class _FakeEmbedder:
        """够用就行：只要求 `_embedder()` 能拿到一个东西 ✓（真崩点在相机 ✗✓）。"""
        def embed(self, *a, **k):
            return [0.0] * 8

    class _FakeFrame:
        image = None

    def _run_concurrent(self, func, n=2):
        import threading
        from unittest import mock
        import forge.camera as cam
        import forge.faces as F

        T = TestFaceToolsHoldTheCamera
        T._FakeSource.peak = T._FakeSource.live = 0
        T._FakeSource.lock = threading.Lock()
        # 每个线程各自延迟一点，制造真正的重叠 ✓
        errs = []
        def call(_i):
            with mock.patch.object(cam, "OpenCvFrameSource", T._FakeSource), \
                 mock.patch.object(cam, "make_detector", lambda *_a, **_k: T._FakeDetector()), \
                 mock.patch.object(F, "make_embedder", lambda *_a, **_k: T._FakeEmbedder()):
                try:
                    func()
                except Exception as e:
                    errs.append("%s: %s" % (type(e).__name__, e))    # ★ 不许静默吞 ✗✓
        ths = [threading.Thread(target=call, args=(i,)) for i in range(n)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout=10)
        T.last_errors = errs
        return T._FakeSource.peak
        ths = [threading.Thread(target=call, args=(i,)) for i in range(n)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout=10)
        return T._FakeSource.peak

    @staticmethod
    def _fn(name):
        """工具在**注册表**里 ✓（不是模块属性 ✗）—— 从那儿取真身 ✓。"""
        from forge import tools as toolmod
        return toolmod.TOOLS[name]["fn"]

    def test_two_face_who_calls_never_overlap(self):
        fn = self._fn("face_who")
        peak = self._run_concurrent(lambda: fn(0))
        self.assertEqual(peak, 1, "两个 face_who 同时开摄像头了 ✗（真机就是这么崩的）：峰值 %d；异常=%s"
                         % (peak, TestFaceToolsHoldTheCamera.last_errors))

    def test_two_face_enroll_calls_never_overlap(self):
        fn = self._fn("face_enroll")
        peak = self._run_concurrent(lambda: fn("甲", samples=1))
        self.assertEqual(peak, 1, "两个 face_enroll 同时开摄像头了 ✗：峰值 %d；异常=%s"
                         % (peak, TestFaceToolsHoldTheCamera.last_errors))
