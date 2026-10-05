"""视觉（摄像头取帧 + 人脸检测）测试 —— 离线、确定性、**不需要真摄像头**。

覆盖三层：

1. **帧源**（`FakeFrameSource` / `capture`）：取帧、帧耗尽、单帧失败重试 —— 全用假帧源，CI 里也真跑
2. **检测**（`StubFaceDetector` / `HaarFaceDetector`）：替身走通链路；Haar 真加载 + 空图**必须返回空列表**
3. **观察与工具**（`observe` / `format_observation` / `annotate` / `encode_image_b64` / `probe_camera` /
   工具注册）：文案结构、不改原帧、JPEG 魔数、**如实汇报可用性**、只读分级

2026-10-03 的取舍：**故意不测真实摄像头**（CI 没有摄像头，测了就是 flaky）。真机验证走
`probe_camera()` 手工确认；所以下面用「固定帧 + 替身检测器」把链路逻辑钉死。
"""
import base64
import importlib.util
import sys
import unittest

try:                          # 无 numpy / cv2 环境：模块仍可被收集，相关用例走 skip
    import numpy as np
except ModuleNotFoundError:   # pragma: no cover
    np = None

HAS_VISION = (importlib.util.find_spec("numpy") is not None
              and importlib.util.find_spec("cv2") is not None)
needs_vision = unittest.skipUnless(HAS_VISION, "视觉测试需要 numpy + opencv（pip install \"handcraft-agent[vision]\"）")


sys.path.insert(0, ".")

from forge.camera import (  # noqa: E402
    CameraError, Face, FakeFrameSource, Frame, HaarFaceDetector, OpenCvFrameSource,
    StubFaceDetector, annotate, available, capture, encode_image_b64, format_observation,
    haar_available, observe, probe_camera,
)


# 光有 cv2 不够：OpenCV 5.x 移除了 Haar（见 camera.haar_available）
HAS_HAAR = HAS_VISION and haar_available()
needs_haar = unittest.skipUnless(HAS_HAAR, "Haar 需要 opencv 4.x（5.x 已移除 Haar 级联）")


class DummyImage:
    """极简假图：只实现链路真正用到的接口（shape / mean / max / min）。

    好处：**没有 numpy 也能测编排逻辑**（取帧 → 检测 → 观察 → 文案），把「需要真依赖」的部分
    缩到最小 —— 只有 Haar 检测和图像编解码那几条才 skip。
    """

    def __init__(self, h=240, w=320, value=100):
        self.shape = (h, w, 3)
        self._v = value

    def mean(self):
        return float(self._v)

    def max(self):
        return self._v + 10

    def min(self):
        return self._v - 10


class EmptyThenFrameSource(FakeFrameSource):
    """前 n 次 read() 返回 None（模拟真实摄像头刚打开时的空帧），之后正常给帧。"""

    def __init__(self, blanks=2, **kw):
        super().__init__(**kw)
        self._blanks = blanks

    def read(self):
        if self._blanks > 0:
            self._blanks -= 1
            return None
        return super().read()


class TestFrameSource(unittest.TestCase):
    def test_fake_source_yields_frames_then_none(self):
        src = FakeFrameSource(frames=[DummyImage(), DummyImage(), DummyImage()], count=3)
        with src:
            frames = []
            while True:
                f = src.read()
                if f is None:
                    break
                frames.append(f)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0].index, 1)
        self.assertEqual(frames[2].index, 3)
        self.assertEqual(frames[0].size, (320, 240))

    def test_fake_source_describe_mentions_no_hardware(self):
        d = FakeFrameSource(count=2, size=(640, 480)).describe()
        self.assertIn("假帧源", d)
        self.assertIn("640x480", d)

    def test_capture_retries_past_empty_frames(self):
        src = EmptyThenFrameSource(blanks=2, frames=[DummyImage()], count=1)
        frame, faces = capture(src, StubFaceDetector())
        self.assertIsNotNone(frame)
        self.assertEqual(faces, [])

    def test_capture_raises_when_nothing_can_be_read(self):
        src = FakeFrameSource(count=0)
        with self.assertRaises(CameraError):
            capture(src)

    def test_opencv_source_requires_open(self):
        src = OpenCvFrameSource()
        if HAS_VISION:
            with self.assertRaises(CameraError):
                src.read()           # 未 open：给的是人话，不是 AttributeError
        else:
            with self.assertRaises(CameraError):
                src.open()


class TestDetector(unittest.TestCase):
    def test_stub_detector_returns_what_was_given(self):
        faces = [Face(10, 20, 30, 40, 1.0), Face(100, 100, 50, 50, 1.0)]
        self.assertEqual(len(StubFaceDetector(faces).detect(DummyImage())), 2)

    @needs_haar
    def test_haar_loads_and_returns_empty_on_blank_image(self):
        """空图**必须**返回空列表 —— 这条是确定性的，能把「检测器永远返回点什么」的错误实现挡掉。"""
        det = HaarFaceDetector()
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(det.detect(blank), [])
        self.assertIn("haar", det.describe())

    @needs_haar
    def test_haar_on_noise_returns_a_list(self):
        """噪点图不保证有人脸，但必须返回 list（不能抛异常、不能返回 None）。"""
        det = HaarFaceDetector()
        rng = np.random.default_rng(20261003)
        noisy = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        out = det.detect(noisy)
        self.assertIsInstance(out, list)
        for f in out:
            self.assertGreater(f.area, 0)

    @needs_vision
    def test_haar_refuses_cleanly_when_api_is_gone(self):
        """OpenCV 5.x 移除了 Haar：必须给「一句人话 + 怎么装回 4.x」，不能是 AttributeError。

        2026-10-03 实测踩到：pip 装到 opencv 5.0，`cv2.CascadeClassifier` 和 `cv2.objdetect` 全没了。
        """
        from unittest import mock
        import forge.camera as cam
        with mock.patch.object(cam, "haar_available", return_value=False):
            with self.assertRaises(CameraError) as ctx:
                cam.HaarFaceDetector()
        self.assertIn("4.x", str(ctx.exception))

    def test_faces_sorted_by_area_desc(self):
        faces = [Face(0, 0, 10, 10), Face(0, 0, 100, 100), Face(0, 0, 50, 50)]
        faces.sort(key=lambda f: f.area, reverse=True)
        self.assertEqual([f.area for f in faces], [10000, 2500, 100])


class TestObservation(unittest.TestCase):
    def _frame(self, **kw):
        return Frame(image=DummyImage(**kw), source="fake")

    def test_observe_shape_and_no_identity_fields(self):
        obs = observe(self._frame(), [])
        self.assertEqual(obs["faces"], 0)
        self.assertEqual(obs["size"], [320, 240])
        for key in ("name", "person", "identity", "embedding"):
            self.assertNotIn(key, obs)          # 观察结果里**不该有**身份字段

    def test_format_no_face(self):
        self.assertIn("没有人脸", format_observation(observe(self._frame(), [])))

    def test_format_one_face(self):
        obs = observe(self._frame(), [Face(10, 20, 30, 40)])
        text = format_observation(obs)
        self.assertIn("1 张人脸", text)
        self.assertIn("(10,20)", text)

    def test_format_many_faces_counts_and_boxes(self):
        obs = observe(self._frame(), [Face(0, 0, 10, 10), Face(50, 50, 20, 20)])
        text = format_observation(obs)
        self.assertIn("2 张人脸", text)
        self.assertIn("#2", text)

    @needs_vision
    def test_annotate_copies_and_does_not_mutate(self):
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        frame = Frame(image=img, source="fake")
        out = annotate(frame, [Face(10, 10, 40, 40)])
        self.assertEqual(int(img.sum()), 0, "原帧必须没被改动")
        self.assertGreater(int(out.sum()), 0, "画了框的副本应该非全黑")

    @needs_vision
    def test_encode_jpeg_has_jpeg_magic(self):
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        raw = base64.b64decode(encode_image_b64(img, "jpeg"))
        self.assertTrue(raw.startswith(b"\xff\xd8"), "JPEG 魔数不对")
        self.assertGreater(len(raw), 100)

    @needs_vision
    def test_encode_png_has_png_magic(self):
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        self.assertTrue(base64.b64decode(encode_image_b64(img, "png")).startswith(b"\x89PNG"))


class TestProbeAndTools(unittest.TestCase):
    def test_available_reports_deps_truthfully(self):
        av = available()
        self.assertEqual(av["opencv"], HAS_VISION)
        self.assertEqual(av["numpy"], np is not None)

    def test_probe_camera_never_raises(self):
        """探测必须**如实汇报**而不是抛异常：没有依赖 / 没有摄像头 / 被占用，都是一种结果。"""
        res = probe_camera(index=9999)          # 不存在的编号：不该炸
        self.assertIsInstance(res, dict)
        self.assertIn("ok", res)
        self.assertIn("reason", res)
        self.assertIn("detail", res)
        if not HAS_VISION:
            self.assertFalse(res["ok"])
            self.assertEqual(res["reason"], "missing_deps")
            self.assertIn("vision", res["detail"])

    def test_tools_registered_only_when_vision_available(self):
        """**不假装有**：没装 opencv/numpy 时，工具列表里不该出现 look / look_image。"""
        from forge.tools import TOOLS, is_write
        if HAS_VISION:
            self.assertIn("look", TOOLS)
            self.assertIn("look_image", TOOLS)
            self.assertFalse(is_write("look"), "look 只读（只看不落盘）")
            self.assertTrue(is_write("look_image"), "look_image 落盘 = 写操作，要单独审批")
        else:
            self.assertNotIn("look", TOOLS)
            self.assertNotIn("look_image", TOOLS)


if __name__ == "__main__":
    unittest.main()
