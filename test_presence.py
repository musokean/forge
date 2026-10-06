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
