"""在场感知（#18 Phase 3）测试 —— 离线、确定性、不需要摄像头/模型。

覆盖四层：

1. **嘴部区域**（`mouth_region`）：形状、边界越界、越界返回 None
2. **嘴部运动**（`MouthMotion`）：静止≈0、动起来>0、平滑、离场清理
3. **跨帧跟踪**（`Tracker`）：同一人移动**不换 id**、走远才换、TTL 回收、面积排序
4. **在场结论**（`PresenceMonitor` / `format_presence`）：识别**降频**（省算力）、名字缓存不闪、
   「谁在说话」的**两条闸门**（过阈值 + 明显领先），拿不准就说「不确定」；以及工具注册与只读分级

时间全部**注入**（`now=`），图像全部**合成** —— 所以本文件是确定性的，CI 里真跑。
**真实场景精度（多人遮挡、侧脸、嘈杂光线）不在覆盖范围内**：那需要摄像头与真人。
"""
import importlib.util
import sys
import unittest

try:
    import numpy as np
except ModuleNotFoundError:   # pragma: no cover
    np = None

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None
needs_numpy = unittest.skipUnless(HAS_NUMPY, "需要 numpy（pip install \"handcraft-agent[vision]\"）")

sys.path.insert(0, ".")

from forge.presence import (  # noqa: E402
    DEFAULT_MIN_MOTION, MouthMotion, PresenceMonitor, Tracker, format_presence,
    mouth_region,
)

from forge.camera import Face  # noqa: E402


def _img(h=120, w=160, v=100):
    return np.full((h, w, 3), v, dtype=np.uint8)


def _talk_frame(base, x0=45, x1=115):
    """在 base 的**嘴部区域**（脸框下三分之一）制造变化 —— 模拟「在说话」。"""
    out = base.copy()
    out[80:105, x0:x1] = 220
    return out


def _face(x=40, y=30, w=80, h=80):
    return Face(x, y, w, h)


def _shifted(img, delta):
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def _steady_shift_sequence(n, delta, h=120, w=160):
    """造 n 帧、每帧相对上一帧整体 +delta —— 稳态的「非嘴动」变化（框平移/曝光微变）。

    逐帧平移而不是单次跳变：`MouthMotion` 的指数平滑会把首帧的 0 拉进来，
    只有**稳态**序列才对应真机那种持续的量级 ✓。
    """
    base = np.full((h, w, 3), 100, dtype=np.int64)
    out = []
    for i in range(n):
        out.append(np.clip(base + delta * i, 0, 255).astype(np.uint8))
    return out


class TestMouthRegion(unittest.TestCase):
    @needs_numpy
    def test_shape_is_downscaled(self):
        small = mouth_region(_img(), _face(), (32, 16))
        self.assertEqual(small.shape, (16, 32, 3))

    @needs_numpy
    def test_picks_lower_third(self):
        img = _img()
        img[80:105, 45:115] = 250          # 只在嘴的位置画亮块
        small = mouth_region(img, _face(), (32, 16))
        self.assertGreater(float(small.mean()), 100)   # 采到了亮块，不是上半张脸

    @needs_numpy
    def test_out_of_frame_box_returns_none(self):
        self.assertIsNone(mouth_region(_img(60, 60), Face(500, 500, 40, 40)))


class TestMouthMotion(unittest.TestCase):
    @needs_numpy
    def test_static_face_scores_zero(self):
        mm, img = MouthMotion(), _img()
        f = _face()
        mm.update(1, mouth_region(img, f, (32, 16)))
        self.assertEqual(mm.update(1, mouth_region(img, f, (32, 16))), 0.0)

    @needs_numpy
    def test_talking_face_scores_positive(self):
        mm, img = MouthMotion(), _img()
        f = _face()
        first = mouth_region(img, f, (32, 16))
        mm.update(1, first)
        score = mm.update(1, mouth_region(_talk_frame(img), f, (32, 16)))
        self.assertGreater(score, DEFAULT_MIN_MOTION)
        self.assertGreater(mm.smooth(1), DEFAULT_MIN_MOTION)

    @needs_numpy
    def test_none_region_is_zero_not_crash(self):
        self.assertEqual(MouthMotion().update(1, None), 0.0)
        self.assertEqual(MouthMotion().smooth(99), 0.0)

    @needs_numpy
    def test_forget_clears_history(self):
        mm, img = MouthMotion(), _img()
        mm.update(7, mouth_region(img, _face(), (32, 16)))
        self.assertIn(7, mm.live_ids())
        mm.forget(7)
        self.assertNotIn(7, mm.live_ids())
        self.assertEqual(mm.smooth(7), 0.0)


class TestTracker(unittest.TestCase):
    @needs_numpy
    def test_two_faces_get_two_ids(self):
        tr = Tracker()
        tracks = tr.update([_face(10, 10), _face(90, 10)], now=0.0)
        self.assertEqual([t.track_id for t in tracks], [1, 2])

    @needs_numpy
    def test_small_move_keeps_the_same_id(self):
        """同一个人的小幅移动**不能换 id** —— 否则名字会跟着换，脸就"跳"了。"""
        tr = Tracker()
        tr.update([_face(40, 30)], now=0.0)
        moved = tr.update([_face(48, 34)], now=0.1)      # 质心移了 ~8px，脸宽 80
        self.assertEqual([t.track_id for t in moved], [1])

    @needs_numpy
    def test_far_jump_becomes_a_new_track(self):
        """跳太远 → 开新 track；**旧 track 不会立刻消失**，而是记一次 misses，等 TTL 到期再回收。

        （写这条用例时我一开始就期望错了 ✗：以为旧的会马上没 —— 实际上"暂留一会儿"正是
        防止侧脸/遮挡导致 id 频繁跳变的手段 ✓。）
        """
        tr = Tracker(max_dist=0.25, ttl=1.0)
        tr.update([_face(10, 10)], now=0.0)
        after = {t.track_id: t for t in tr.update([_face(120, 10)], now=0.1)}
        self.assertIn(2, after, "跳远的人应该拿到新 id")
        self.assertEqual(after[1].misses, 1, "旧 track 应先记一次未跟上")
        self.assertEqual(len(tr.update([], now=2.0)), 0, "超过 TTL 后旧 track 才回收")

    @needs_numpy
    def test_ttl_expiry_removes_track(self):
        tr = Tracker(ttl=1.0)
        tr.update([_face(10, 10)], now=0.0)
        self.assertEqual(len(tr.update([_face(10, 10)], now=0.5)), 1)     # 还在
        self.assertEqual(len(tr.update([], now=2.0)), 0)                  # 超过 TTL → 回收
        self.assertEqual(len(tr.active()), 0)

    @needs_numpy
    def test_tracks_sorted_by_area_desc(self):
        tr = Tracker()
        tracks = tr.update([_face(10, 10, 40, 40), _face(90, 10, 90, 90)], now=0.0)
        self.assertGreaterEqual(tracks[0].box.area, tracks[1].box.area)

    @needs_numpy
    def test_as_dict_has_no_identity_leak_beyond_name(self):
        tr = Tracker()
        d = tr.update([_face()], now=0.0)[0].as_dict()
        self.assertEqual(d["name"], "未登记#1")
        self.assertIn("mouth", d)
        self.assertNotIn("image", d)


class TestPresenceMonitor(unittest.TestCase):
    @needs_numpy
    def test_recognition_is_throttled(self):
        """**降频**：15 帧、每 5 帧认一次 → 最多 3 次识别（省算力，且这是可断言的 ✓）。"""
        calls = []
        mon = PresenceMonitor(recognizer=lambda img, box: (calls.append(1), ("老王", 0.9))[1],
                              recognize_every=5)
        img = _img()
        for _ in range(15):
            mon.observe(img, [_face()], now=0.0)
        self.assertLessEqual(mon.recognize_calls, 3)
        self.assertEqual(len(calls), mon.recognize_calls)

    @needs_numpy
    def test_name_is_cached_when_recognition_fails(self):
        """某一帧没认出来（侧脸/模糊）时**保留**上次的名字 —— 否则名字会一闪一闪 ✗。"""
        state = {"ok": True}
        mon = PresenceMonitor(recognizer=lambda img, box: (("老王", 0.9) if state["ok"] else ("", 0.2)),
                              recognize_every=1)
        img = _img()
        mon.observe(img, [_face()], now=0.0)
        state["ok"] = False
        tracks = mon.observe(img, [_face()], now=0.1)
        self.assertEqual(tracks[0].name, "老王")

    @needs_numpy
    def test_without_recognizer_still_tracks(self):
        mon = PresenceMonitor(recognizer=None)
        tracks = mon.observe(_img(), [_face()], now=0.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].name, "")

    @needs_numpy
    def test_snapshot_picks_the_moving_mouth(self):
        mon = PresenceMonitor(recognizer=None, min_motion=0.005)
        img = _img()
        faces = [_face(10, 10), _face(90, 10)]
        mon.observe(img, faces, now=0.0)
        # 只让**第一张脸**的嘴部动起来
        moved = img.copy()
        moved[80:105, 15:75] = 240
        mon.observe(moved, faces, now=0.1)
        snap = mon.snapshot()
        self.assertIsNotNone(snap["speaking"])
        self.assertEqual(snap["speaking"], "未登记#1")

    @needs_numpy
    def test_snapshot_unknown_when_nobody_moves(self):
        mon = PresenceMonitor(recognizer=None, min_motion=0.05)
        img = _img()
        mon.observe(img, [_face()], now=0.0)
        mon.observe(img, [_face()], now=0.1)
        snap = mon.snapshot()
        self.assertIsNone(snap["speaking"])
        self.assertIn("低于阈值", snap["reason"])

    @needs_numpy
    def test_snapshot_unknown_when_two_talk_alike(self):
        """两个人嘴动分接近 → 说「分不清」，不硬指一个 ✓。"""
        mon = PresenceMonitor(recognizer=None, min_motion=0.001, ratio=3.0)
        img = _img()
        faces = [_face(10, 10), _face(90, 10)]
        mon.observe(img, faces, now=0.0)
        moved = img.copy()
        moved[80:105, 15:75] = 240           # 两张脸都动，幅度接近
        moved[80:105, 95:155] = 245
        mon.observe(moved, faces, now=0.1)
        snap = mon.snapshot()
        self.assertIsNone(snap["speaking"])
        self.assertIn("分不清", snap["reason"])

    @needs_numpy
    def test_audio_gate_says_nobody_is_speaking(self):
        """**本轮的核心修复** ✓✓：音频说「没人说话」→ 即使画面噪声给出了分数，也必须回「没人说话」。

        （实测：安静时嘴动分 0.0159~0.0223，与说话时重叠 ✗ —— 所以视觉那一半根本不该单独下结论 ✓。）
        """
        mon = PresenceMonitor(recognizer=None, min_motion=0.001, audio=lambda: False)
        img = _img()
        mon.observe(img, [_face()], now=0.0)
        mon.observe(_talk_frame(img), [_face()], now=0.1)      # 视觉上「在动嘴」
        snap = mon.snapshot()
        self.assertIsNone(snap["speaking"], "音频说没人说话时不得判定有人在说话 ✗")
        self.assertIn("音频判定", snap["reason"])
        self.assertEqual(snap["mode"], "audio_gated")
        self.assertIn("没人在说话", format_presence(snap, 3.0))

    @needs_numpy
    def test_audio_gate_uses_vision_only_for_who(self):
        """音频说「有人在说话」→ 视觉只负责「是谁」✓。"""
        mon = PresenceMonitor(recognizer=None, min_motion=0.001, audio=lambda: True)
        img = _img()
        mon.observe(img, [_face()], now=0.0)
        mon.observe(_talk_frame(img), [_face()], now=0.1)
        snap = mon.snapshot()
        self.assertEqual(snap["speaking"], "未登记#1")
        self.assertEqual(snap["mode"], "audio_gated")
        self.assertIs(snap["audio"], True)
        self.assertIn("音频判定有人在说话", format_presence(snap, 3.0))

    @needs_numpy
    def test_audio_speech_is_not_vetoed_by_the_motion_threshold(self):
        """**本轮真机踩到的 bug** ✗✓：音频说"有人说话"时，视觉的嘴动阈值**不许否否决**。

        真机现场：音频报"有人说话"的那 2 秒，嘴动分 0.0281 < min_motion 0.03 → 工具回了
        "没人说话" ✗ —— 门控装反了。音频（可靠）定了"有没有"，视觉（不可靠）只该定"是谁"✓。
        """
        mon = PresenceMonitor(recognizer=None, min_motion=0.03, audio=lambda: True)
        img = _img()
        mon.observe(img, [_face()], now=0.0)
        # 幅度调在纯视觉阈值以下（≈0.0196 < 0.03），但音频说有人在说话
        for i in range(6):
            mon.observe(_shifted(img, 5 * (i + 1)), [_face()], now=0.1 * (i + 1))
        snap = mon.snapshot()
        self.assertEqual(snap["speaking"], "未登记#1",
                         "音频说有人在说话时，画面里只有一个人 → 就该归给他，不许被嘴动阈值否决 ✗")
        self.assertIn("音频判定有人在说话", snap["reason"])

    @needs_numpy
    def test_audio_speech_with_several_faces_prefers_the_moving_one(self):
        mon = PresenceMonitor(recognizer=None, min_motion=0.001, audio=lambda: True)
        img = _img()
        faces = [_face(10, 10), _face(90, 10)]
        mon.observe(img, faces, now=0.0)
        # 变化**只落在第一张脸的框内**（x 15~75 ⊂ 脸#1 的 10~90 ✓）——
        # 一开始我写成 45~115，横跨到了脸#2，于是"分不清"反而是**正确**行为 ✓✓
        mon.observe(_talk_frame(img, 15, 75), faces, now=0.1)
        snap = mon.snapshot()
        self.assertEqual(snap["speaking"], "未登记#1", "多张脸时用嘴动分挑说话的那个 ✓")

    @needs_numpy
    def test_audio_speech_but_cannot_tell_who(self):
        """音频说有人说话，但画面多人且分不出嘴动 → 明说"不硬指" ✓（不许瞎指一个 ✗）。"""
        mon = PresenceMonitor(recognizer=None, min_motion=0.001, ratio=3.0, audio=lambda: True)
        img = _img()
        faces = [_face(10, 10), _face(90, 10)]
        mon.observe(img, faces, now=0.0)
        moved = img.copy()
        moved[80:105, 15:75] = 240
        moved[80:105, 95:155] = 245          # 两张脸都在动、幅度接近
        mon.observe(moved, faces, now=0.1)
        snap = mon.snapshot()
        self.assertIsNone(snap["speaking"])
        self.assertIn("不硬指", snap["reason"])

    @needs_numpy
    def test_a_broken_gate_falls_back_instead_of_crashing(self):
        """门控自己抛异常 → 退回纯视觉，不能把整个查询弄挂 ✓。"""
        def boom():
            raise RuntimeError("mic died")

        mon = PresenceMonitor(recognizer=None, min_motion=0.001, audio=boom)
        mon.observe(_img(), [_face()], now=0.0)
        snap = mon.snapshot()
        self.assertEqual(snap["mode"], "motion_only")
        self.assertIsNone(snap["audio"])

    @needs_numpy
    def test_without_audio_it_never_claims_speech(self):
        """没音频时**不许说「正在说话」** ✗✓ —— 只能说「嘴动最明显」。这是诚实性守卫。"""
        mon = PresenceMonitor(recognizer=None, min_motion=0.001)       # 无 audio
        img = _img()
        mon.observe(img, [_face()], now=0.0)
        mon.observe(_talk_frame(img), [_face()], now=0.1)
        snap = mon.snapshot()
        self.assertEqual(snap["mode"], "motion_only")
        text = format_presence(snap, 3.0)
        self.assertIn("不能确认在说话", text)
        self.assertNotIn("正在说话", text, "没有音频就不许声称「在说话」 ✗")

    @needs_numpy
    def test_default_does_not_fire_on_the_measured_noise_floor(self):
        """**把真机实测结论钉成回归测试** ✓✓。

        实测：安静时嘴动分中位 0.0159 / max 0.0223，说话时中位 0.0199 —— 两者重叠 ✗。
        所以默认阈值必须**高于这个噪声带**，否则会在人闭嘴时误报（旧默认 0.015 就会 ✗）。
        这里合成一段幅度恰好等于噪声带的帧间变化（每像素 +4 → 4/255 ÷ 3 通道 ≈ 0.0157），
        断言：**默认参数下不得判定为「在说话」** ✓。
        """
        frames = _steady_shift_sequence(8, 5)           # 每帧 +5 → ≈0.0196，正落在实测重叠区 ✓
        mon = PresenceMonitor(recognizer=None)          # 用**默认** min_motion ✓
        for t in range(len(frames)):
            mon.observe(frames[t], [_face()], now=t * 0.1)
        snap = mon.snapshot()
        self.assertIsNone(snap["speaking"], f"重叠区幅度不该被判成说话：{snap}")
        self.assertGreater(mon.min_motion, 0.0223, "默认必须高于实测噪声 max（0.0223）")
        # 反证：旧默认 0.015 会误报 —— 说明这个测试确实在测那条边界 ✓（这才是要修的 bug）
        old = PresenceMonitor(recognizer=None, min_motion=0.015)
        for t in range(len(frames)):
            old.observe(frames[t], [_face()], now=t * 0.1)
        self.assertIsNotNone(old.snapshot()["speaking"], "旧默认 0.015 应当会误报（这正是要修的 bug）")

    @needs_numpy
    def test_snapshot_empty_scene(self):
        snap = PresenceMonitor(recognizer=None).snapshot()
        self.assertEqual(snap["people"], [])
        self.assertIn("没有人", snap["reason"])

    @needs_numpy
    def test_format_text_branches(self):
        """四种措辞都要对，而且**措辞必须和证据强度匹配** ✓✓（这是诚实性的一部分）。"""
        self.assertIn("没有人", format_presence({"people": [], "speaking": None, "reason": "画面里没有人"}, 1.0))
        # 有音频门控 + 判定有人说话 → 可以说「正在说话」✓
        gated = {"people": [{"name": "老王"}], "speaking": "老王", "score": 0.05, "audio": True, "reason": ""}
        self.assertIn("正在说话：老王", format_presence(gated, 3.0))
        # 没有音频 → **只说「嘴动最明显」，不许说「正在说话」** ✗✓
        motion = {"people": [{"name": "老王"}], "speaking": "老王", "score": 0.05, "audio": None, "reason": ""}
        text = format_presence(motion, 3.0)
        self.assertIn("嘴动最明显", text)
        self.assertNotIn("正在说话", text, "没音频还声称「在说话」就是过度宣称 ✗")
        # 音频说没人说话 → 直接回没人说话 ✓
        quiet = {"people": [{"name": "老王"}], "speaking": None, "audio": False,
                 "reason": "音频判定：此刻没人在说话"}
        self.assertIn("此刻没人在说话", format_presence(quiet, 3.0))
        # 拿不准 → 不确定 ✓
        unknown = {"people": [{"name": "老王"}], "speaking": None, "audio": True, "reason": "没人在明显动嘴"}
        self.assertIn("不确定", format_presence(unknown, 3.0))


class TestTools(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMPY and HAS_CV2, "工具注册需要 numpy + opencv")
    def test_who_is_speaking_registered_readonly(self):
        from forge.tools import TOOLS, is_write
        self.assertIn("who_is_speaking", TOOLS)
        self.assertFalse(is_write("who_is_speaking"), "只看不动嘴，是只读操作")


if __name__ == "__main__":
    unittest.main()


@needs_numpy
class TestFormatAmbient(unittest.TestCase):
    """#18 紧凑现场措辞 + 探针的回答前措辞（含「只有一个人就是他」的规则 ✓）。"""

    def test_no_people(self):
        from forge.presence import format_ambient
        self.assertEqual(format_ambient({"people": []}), "画面里没有人。")

    def test_names_and_scores(self):
        from forge.presence import format_ambient
        s = format_ambient({"people": [{"name": "满仓", "score": 0.914},
                                       {"name": "", "track_id": 3, "score": 0.5}]})
        self.assertIn("在场 2 人", s)
        self.assertIn("满仓（0.91）", s)
        self.assertIn("未登记#3", s, "没名字只报「未登记#N」✓ 不硬编名字 ✗")

    def _probe_with(self, people):
        from forge.presence import PresenceProbe

        class _M:
            def snapshot(self):
                return {"people": people}
        p = PresenceProbe.__new__(PresenceProbe)     # 不启线程/不开摄像头
        p._mon = _M()
        return p

    def test_one_person_is_the_speaker(self):
        p = self._probe_with([{"name": "满仓", "score": 0.91}])
        s = p.ambient_for_answer()
        self.assertIn("刚才说话的是 满仓", s)

    def test_several_people_refuse_to_point(self):
        p = self._probe_with([{"name": "满仓", "score": 0.91}, {"name": "小李", "score": 0.88}])
        s = p.ambient_for_answer()
        self.assertIn("分不出", s, "多人时要说清「分不出来」✓")
        self.assertIn("2 人", s)
        self.assertNotIn("刚才说话的是 小李", s, "多人时不许硬指 ✗")

    def test_speaker_off_camera(self):
        p = self._probe_with([])
        self.assertIn("不在画面里", p.ambient_for_answer())

class TestProbeUsesSameRecipeAsEnrolment(unittest.TestCase):
    """★ 探针识别必须走**和登记完全相同**的管线 ✓✓。

    2026-10-06 真机踩到 ✗：探针自己拼 `embedder.embed(crop_face(...))`（裁剪 ✗），登记走
    `embed_face`（**对齐** ✓）—— 同一张脸同一帧，两条路的向量余弦只有 **0.537**，匹配分从
    **0.762 掉到 0.448**（贴着 0.36 阈值 → 真机上就表现为「现场全是未登记」✗）。
    这条测试**不需要摄像头** ✓：假检测器给关键点、假提取器**记录被调用了哪个方法** ✓。
    """

    def setUp(self):
        if not HAS_NUMPY:
            self.skipTest("需要 numpy")

    def test_recognizer_prefers_aligned_path_when_landmarks_available(self):
        from forge.presence import PresenceProbe

        calls = []

        class FakeFace:
            x, y, w, h, score = 10, 10, 80, 80, 0.95

            @property
            def area(self):
                return self.w * self.h

        class FakeDet:
            def detect(self, image):
                return [FakeFace()]

            def detect_with_landmarks(self, image):
                return [(FakeFace(), "ROW")]          # ← 有关键点 → 必须走对齐 ✓

        class FakeEmb:
            def embed(self, crop):
                calls.append("embed")
                return [0.0] * 8

            def embed_aligned(self, image, row):
                calls.append("embed_aligned")
                assert row == "ROW", "必须把关键点行原样交给 embed_aligned ✓"
                return [1.0] + [0.0] * 7

        class FakeMatch:
            name, score, unknown = "满仓", 0.9, False

        class FakeStore:
            def match(self, vec, **kw):
                assert vec[0] == 1.0, "拿到的必须是 embed_aligned 的向量 ✓"
                return FakeMatch()

        probe = PresenceProbe(interval_s=1.0, store=FakeStore(), embedder=FakeEmb(), detector=FakeDet())
        rec = probe._make_recognizer()
        nm, sc = rec(object(), FakeFace())
        self.assertEqual(calls, ["embed_aligned"], "探针必须走对齐路径，绝不能退回裁剪 ✗")
        self.assertEqual(nm, "满仓")
        self.assertGreater(sc, 0.0)
        self.assertEqual(probe.first_error, "", "不该有静默失败 ✓")

    def test_recognizer_reports_failure_reason_once(self):
        """识别失败必须**报出原因**（只报一次）✓ —— 静默吞掉 ⇒ 现场全是「未登记」而无人知道为什么 ✗✓。"""
        from forge.presence import PresenceProbe

        class BoomEmb:
            def embed(self, crop):
                raise RuntimeError("模型加载失败（模拟）")

        class FakeDet:
            def detect(self, image):
                return []

        class FakeFace:
            x, y, w, h, score = 0, 0, 50, 50, 0.9

            @property
            def area(self):
                return 2500

        probe = PresenceProbe(interval_s=1.0, store=object(), embedder=BoomEmb(), detector=FakeDet())
        rec = probe._make_recognizer()
        img = np.zeros((48, 48, 3), dtype=np.uint8)          # 假图要有 .shape ✓，否则异常发生在更早一步 ✗
        nm1, sc1 = rec(img, FakeFace())
        nm2, _ = rec(img, FakeFace())
        self.assertEqual((nm1, sc1), ("", 0.0))
        self.assertEqual(nm2, "")
        self.assertIn("模型加载失败（模拟）", probe.first_error, "失败原因必须被记下来 ✓")

class TestAvSpeaker(unittest.TestCase):
    """★ 用「音频包络 × 每人嘴动」的互相关判断**谁在说话** ✓✓（两人场景的关键缺口 ✓）。

    不需要摄像头、不需要 numpy、不需要真人 ✓ —— 全用**合成时序**，因为这里要验的是**判据**：
      · 与声音同步的人胜出 ✓（并允许**嘴唇领先声音** ✓：真实约 100~200ms ✓）
      · 没人同步 → **说不清** ✓
      · **两个人都在同步**（= 同时在说）→ **说不清** ✓✓（这条就是 margin 的意义 ✓）
      · 样本太少 → 说不清 ✓
    另附**反证**：把 margin 去掉 → 同时在说的场景会**乱指一个** ✗ → 说明 margin 真在起作用 ✓。
    """

    def _series(self, av, seed=1):
        """造一段：音频包络 `env` 的序列 + 两个 track 的嘴动 ✓。"""
        import random
        rng = random.Random(seed)
        n = len(av)
        for i, a in enumerate(av):
            av_ts = i * 0.1
            av_val = a
        return n

    def _feed(self, av, mouth_a, mouth_b, dt=0.1, lead=0.15):
        """audio 与 mouth 分开喂 ✓；mouth_a 与**早 lead 秒的音频**同步（= 嘴唇领先 ✓）。"""
        from forge.presence import AvSpeaker
        sp = AvSpeaker(window_s=5.0, min_samples=6, min_corr=0.35, margin=0.15)
        for i, a in enumerate(av):
            sp.on_audio(i * dt, a)
        for i in range(len(av)):
            ti = i * dt
            # 采样时若嘴唇领先 lead，则「t 时刻的嘴」对应「t-lead 时刻的声音」 →
            # 在 pairs(lag) 里表现为 best_lag = +lead ✓
            sp.on_mouth(1, ti, mouth_a[i])
            sp.on_mouth(2, ti, mouth_b[i])
        return sp

    def test_speaker_wins_with_lip_lead(self):
        """A 的嘴动跟着声音（领先 0.15s ✓），B 是无关动作 → 应判 A ✓✓，且最佳延迟 ≈ +0.15 ✓。"""
        import math
        env = [0.5 + 0.5 * math.sin(i / 2.0) for i in range(50)]          # 有起伏的包络 ✓
        mouth_a = [0.5 + 0.5 * math.sin((i - 1.5) / 2.0) for i in range(50)]   # 领先 1.5 步 = 0.15s ✓
        mouth_b = [0.5 + 0.5 * math.cos(i / 3.7) for i in range(50)]      # 无关动作 ✓
        sp = self._feed(env, mouth_a, mouth_b)
        d = sp.decide()
        self.assertEqual(d["track_id"], 1, "应判为 A（他的嘴动与声音同步 ✓）：%s" % d)
        self.assertGreater(d["corr"], 0.8, "相关系数应很高 ✓：%s" % d)
        self.assertAlmostEqual(d["lag"], 0.15, delta=0.06, msg="最佳延迟应≈+0.15s（嘴唇领先 ✓）：%s" % d)

    def test_no_sync_admits_it(self):
        """两个人都在乱动 → **说不清** ✓（不许硬指 ✗）。"""
        import math
        env = [0.5 + 0.5 * math.sin(i / 2.0) for i in range(50)]
        mouth_a = [0.5 + 0.5 * math.cos(i / 4.3) for i in range(50)]
        mouth_b = [0.5 + 0.5 * math.sin(i / 5.9) for i in range(50)]
        sp = self._feed(env, mouth_a, mouth_b)
        d = sp.decide()
        self.assertIsNone(d["track_id"], "没人同步就该说不清 ✓：%s" % d)
        self.assertIn("同步", d["reason"])

    def test_two_speakers_at_once_is_ambiguous(self):
        """**两个人同时在说**（两边的嘴动都跟着声音）→ 必须**说不清** ✓✓，不能挑一个 ✗。"""
        import math
        env = [0.5 + 0.5 * math.sin(i / 2.0) for i in range(50)]
        mouth_a = [0.5 + 0.5 * math.sin((i - 1.5) / 2.0) for i in range(50)]
        mouth_b = [0.5 + 0.5 * math.sin((i - 1.5) / 2.0) + 0.01 * math.sin(i) for i in range(50)]
        sp = self._feed(env, mouth_a, mouth_b)
        d = sp.decide()
        self.assertIsNone(d["track_id"], "同时在说就是分不出，不许硬指 ✗：%s" % d)

    def test_margin_is_what_refuses(self):
        """★ **反证** ✓：把 margin 拆掉 → 上面那个「同时在说」的场景就会**乱指一个** ✗。

        这条测试保证 `margin` 不是摆设 ✓（去掉它，上面的用例就会失败 ✓）。
        """
        import math
        from forge.presence import AvSpeaker
        env = [0.5 + 0.5 * math.sin(i / 2.0) for i in range(50)]
        mouth_a = [0.5 + 0.5 * math.sin((i - 1.5) / 2.0) for i in range(50)]
        mouth_b = [0.5 + 0.5 * math.sin((i - 1.5) / 2.0) + 0.01 * math.sin(i) for i in range(50)]
        no_margin = AvSpeaker(window_s=5.0, min_samples=6, min_corr=0.35, margin=-1.0)
        for i, a in enumerate(env):
            no_margin.on_audio(i * 0.1, a)
        for i in range(len(env)):
            no_margin.on_mouth(1, i * 0.1, mouth_a[i])
            no_margin.on_mouth(2, i * 0.1, mouth_b[i])
        self.assertIsNotNone(no_margin.decide()["track_id"],
                             "去掉 margin 后它会乱指一个 ✗ —— 这正是 margin 要拦住的 ✓")

    def test_too_few_samples_is_honest(self):
        from forge.presence import AvSpeaker
        sp = AvSpeaker(window_s=5.0, min_samples=6)
        sp.on_audio(0.0, 0.3)
        sp.on_audio(0.1, 0.5)
        sp.on_mouth(1, 0.0, 0.4)
        d = sp.decide()
        self.assertIsNone(d["track_id"])
        self.assertIn("样本不足", d["reason"])

    def test_window_prunes_old_samples(self):
        """窗口外的旧样本必须被丢 ✓（否则会拿几分钟前的动作来相关 ✗）。"""
        from forge.presence import AvSpeaker
        sp = AvSpeaker(window_s=1.0, min_samples=3)
        for i in range(40):
            sp.on_audio(i * 0.1, 0.5 + 0.5 * (i % 2))
            sp.on_mouth(1, i * 0.1, 0.5 + 0.5 * (i % 2))
        self.assertLessEqual(len(sp.pairs(1)), 12, "1.0s 窗口 @0.1s → 最多 ~11 个样本 ✓")
        self.assertGreaterEqual(len(sp.pairs(1)), 6)

class TestSpeechBurst(unittest.TestCase):
    """★ 突发采样：**有人说话才开摄像头** ✓✓（不需要麦克风/摄像头/真人 ✓）。

    验四件事：① 没人说话时**相机根本不开** ✓（隐私 ✓）；② 说话时连开、按 fps 采帧、把音频包络与
    每人嘴动喂进互相关 ✓；③ 静音 `stop_silence_s` 后**关相机并给出判定** ✓；④ 相机**确实被释放** ✓。
    """

    class FakeGate:
        def __init__(self, clock):
            self.clock = clock
            self.loud = True
            self.series = []
        def speaking(self):
            return self.loud
        def loud_ms(self):
            return 400 if self.loud else 0
        def energy_series(self, since_mono=None):
            return [(m, r) for m, r in self.series if since_mono is None or m > since_mono]

    class FakeSource:
        def __init__(self):
            self.opened = 0
            self.closed = 0
        def open(self):
            self.opened += 1
        def close(self):
            self.closed += 1
        def read(self):
            return "FRAME"

    class FakeDetector:
        def detect(self, image):
            return ["FACE"]

    class FakeTrack:
        def __init__(self, tid, mouth, name=""):
            self.track_id = tid
            self.mouth = mouth
            self.name = name
            self.box = None

    class FakeMonitor:
        """按**预设波形**返回两个 track 的嘴动分 ✓（1 号与音频同相 ✓，2 号反相 ✓）。"""
        def __init__(self, clock, gate):
            self.clock = clock
            self.gate = gate
            self.calls = 0
            class T:
                def active(self_inner):
                    return []
            self.tracker = T()
        def observe(self, frame, faces, now=None):
            self.calls += 1
            import math
            a = 0.5 + 0.5 * math.sin(self.calls / 2.0)          # 与音频同形 ✓
            b = 0.5 + 0.5 * math.cos(self.calls / 3.7)          # 无关 ✓
            return [TestSpeechBurst.FakeTrack(1, a, "满仓"), TestSpeechBurst.FakeTrack(2, b, "翠花")]

    def _rig(self):
        import math
        now = {"t": 1000.0}
        clock = lambda: now["t"]
        from forge.presence import SpeechBurst
        gate = TestSpeechBurst.FakeGate(clock)
        src = TestSpeechBurst.FakeSource()
        mon = TestSpeechBurst.FakeMonitor(clock, gate)
        burst = SpeechBurst(mon, gate, source=src, detector=TestSpeechBurst.FakeDetector(), fps=10.0,
                            stop_silence_s=0.6, max_burst_s=5.0, clock=clock)
        return burst, gate, src, mon, now

    def test_camera_stays_shut_when_nobody_speaks(self):
        """① 没人说话 → **相机一次都不开** ✓✓（这就是隐私保证 ✓）。"""
        burst, gate, src, mon, now = self._rig()
        gate.loud = False
        for _ in range(50):
            now["t"] += 0.1
            self.assertIsNone(burst.step())
        self.assertEqual(src.opened, 0, "没人说话时绝不能开相机 ✗")
        self.assertEqual(burst.bursts, 0)
        self.assertFalse(burst.bursting)

    def test_burst_opens_reads_decides_and_releases(self):
        """②③④ 说话 → 连开采样 → 静音后**关相机 + 判定出与声音同步的那位** ✓✓。"""
        import math
        burst, gate, src, mon, now = self._rig()
        gate.loud = True
        # 说话期间：音频包络与「1 号的嘴动」同相（外加 0.15s 嘴唇领先 ✓）
        for i in range(30):
            t = 1000.0 + i * 0.1
            gate.series.append((t, 0.5 + 0.5 * math.sin(i / 2.0)))
            now["t"] = t
            burst.step()
        self.assertTrue(burst.bursting, "有人说话时应当处于突发状态 ✓")
        self.assertEqual(src.opened, 1, "连开一次 ✓（不是每帧开关 ✗）")
        got = mon.calls
        self.assertGreaterEqual(got, 5, "应当连续采到若干帧 ✓（fps 限速 ✓）：%d" % got)
        # 静音 → 停
        gate.loud = False
        out = None
        for _ in range(20):
            now["t"] += 0.1
            out = burst.step()
            if out is not None:
                break
        self.assertIsNotNone(out, "静音 0.6s 后应当结束并返回判定 ✓")
        self.assertEqual(out["track_id"], 1, "应判为与声音同步的 1 号 ✓：%s" % out)
        self.assertGreater(out["corr"], 0.5)
        self.assertEqual(src.closed, 1, "结束后**必须关掉相机** ✓✓")
        self.assertFalse(burst.bursting)

    def test_long_burst_is_capped(self):
        """一直有声音也不能永远占着相机 ✓（max_burst_s 封顶 ✓）。"""
        burst, gate, src, mon, now = self._rig()
        gate.loud = True
        out = None
        for _ in range(80):                      # 8 秒 > max_burst_s=5 ✓
            now["t"] += 0.1
            out = burst.step()
            if out is not None:
                break
        self.assertIsNotNone(out, "超过 max_burst_s 必须收尾 ✓")
        self.assertEqual(src.closed, 1, "封顶后也要关相机 ✓")

    def test_read_failure_is_reported_not_swallowed(self):
        """取帧失败**要说话** ✓（今天栽过：静默吞掉 ⇒ 现场数据空而无从得知 ✗✓）。"""
        burst, gate, src, mon, now = self._rig()
        def boom():
            raise RuntimeError("模拟取帧失败")
        src.read = boom
        gate.loud = True
        now["t"] += 0.1
        burst.step()                      # 第一步只负责"起突发" ✓
        now["t"] += 0.1
        burst.step()                      # 第二步才会读帧 ✓
        self.assertIn("模拟取帧失败", burst.last_error, "失败原因必须被记下 ✓")

class TestAmbientCarriesAvVerdict(unittest.TestCase):
    """AV 判定要进**注入给模型的那句话** ✓✓；判不准时照样说「分不出」✓。"""

    def _probe(self, av):
        from forge.presence import PresenceProbe
        pr = PresenceProbe.__new__(PresenceProbe)          # 不起线程、不开相机 ✓
        pr._mon = type("M", (), {"snapshot": lambda self=None: {"people": []}})()
        pr._av_result = av
        return pr

    def test_names_the_speaker_when_av_is_confident(self):
        pr = self._probe(lambda: {"track_id": 2, "corr": 0.81, "names": {2: "翠花"}, "reason": "与声音同步"})
        line = pr._av_line()
        self.assertIn("翠花", line)
        self.assertIn("0.81", line)

    def test_admits_when_av_cannot_tell(self):
        pr = self._probe(lambda: {"track_id": None, "corr": 0.4,
                                  "reason": "两个人都在动、相关也接近（0.72 vs 0.70）→ 分不出"})
        line = pr._av_line()
        self.assertIn("分不出", line)
        self.assertNotIn("满仓", line)

    def test_no_av_means_no_claim(self):
        self.assertEqual(self._probe(None)._av_line(), "")
        self.assertEqual(self._probe(lambda: {})._av_line(), "")
        self.assertEqual(self._probe(lambda: (_ for _ in ()).throw(RuntimeError("坏")))._av_line(), "")

class TestSpeechBurstPausesWhileAgentTalks(unittest.TestCase):
    """★ 它自己在说话时必须**暂停**突发采样 ✓✓。

    否则：半双工只闭了**主循环**的麦 ✗，突发采样用的是**独立**音频流 ✓ → 会把 forge 自己的声音
    当成「有人在说」✗ → 采到的嘴动与声音不同步 → 判定必错 ✗✓（甚至把它自己判成说话人 ✗）。
    """

    def test_paused_hook_stops_and_releases(self):
        now = {"t": 500.0}
        clock = lambda: now["t"]
        from forge.presence import SpeechBurst

        class G:
            loud = False
            def speaking(self):
                return False
            def loud_ms(self):
                return 0
            def energy_series(self, since_mono=None):
                return []

        pause = {"v": False}
        src = type("S", (), {"opened": 0, "closed": 0,
                             "open": lambda self: setattr(self, "opened", self.opened + 1),
                             "close": lambda self: setattr(self, "closed", self.closed + 1),
                             "read": lambda self: "F"})()

        class M:
            class T:
                def active(self):
                    return []
            tracker = T()
            def observe(self, frame, faces, now=None):
                return []

        b = SpeechBurst(M(), G(), source=src, detector=None, clock=clock, paused=lambda: pause["v"])
        b._open()                      # 模拟"已经在采样"✓
        b._bursting = True
        pause["v"] = True              # 它开始说话了 ✓
        self.assertIsNone(b.step())
        self.assertFalse(b.bursting, "说话时必须停下 ✓")
        self.assertEqual(src.closed, 1, "并且**立刻放掉相机** ✓✓")
