"""摄像头取帧 + 人脸检测（#18 Phase 1）—— 让 Agent 能「看一眼」。

这一层只回答两个问题：**画面里有没有人、人脸在哪**（检测）。
「这是谁」是下一步（识别 + 身份库），本模块**不做身份识别、不做人脸比对、不存原图**。

设计原则（和语音那套一致）：

  · **可注入假帧源** —— `FakeFrameSource` 让整条链路在没有摄像头的环境（含 CI）里也能测；
  · **不假装有** —— 没装 cv2、或摄像头打不开时，给一句人话 + 处置办法，而不是丢一个 ImportError 栈；
  · **隐私默认最保守** —— `look()` 只回数字（尺寸 / 人脸数 / 框位置），**不落盘**；
    要留图必须显式调 `look_image()`（写操作，会被只读分级单独看待）。

隐私提醒：人脸属于**敏感个人信息**（PIPL / GDPR 均按特殊类别处理）。本模块只在内存里处理帧，
不做身份识别；将来做识别也要遵循「只存特征向量、可删除、需本人同意」。
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

try:                          # 核心安装不含 numpy / cv2（只在 [vision] extra 里）：
    import numpy as np        # 模块仍要能导入，真正用到时再由 _require_*() 给出明确提示
except ModuleNotFoundError:   # pragma: no cover
    np = None

try:
    import cv2
except ModuleNotFoundError:   # pragma: no cover
    cv2 = None

__all__ = [
    "CameraError",
    "grab_frame",
    "haar_available",
    "Face",
    "Frame",
    "FrameSource",
    "OpenCvFrameSource",
    "FakeFrameSource",
    "FaceDetector",
    "HaarFaceDetector",
    "StubFaceDetector",
    "annotate",
    "encode_image_b64",
    "probe_camera",
    "available",
]

# Haar 级联文件随 opencv 一起发布（不需要联网下载模型）——这是选它做 Phase 1 的主要原因。
_CASCADE_FILE = "haarcascade_frontalface_default.xml"


class CameraError(RuntimeError):
    """摄像头/视觉链路不可用。**带了怎么办**，不是裸异常。"""


def _require_vision():
    if np is None or cv2 is None:
        raise CameraError(
            "摄像头功能需要 numpy + opencv：pip install \"handcraft-agent[vision]\""
            "（或 pip install numpy opencv-python-headless）"
        )


# ══════════════════════════════════════════════════════════════════
# 数据结构


@dataclass
class Face:
    """检测到的一张脸（**不是身份** —— 只是「这里有一张脸」）。"""

    x: int
    y: int
    w: int
    h: int
    score: float = 0.0

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "score": round(self.score, 3)}


@dataclass
class Frame:
    """一帧画面。`image` 是 BGR ndarray（OpenCV 习惯），**只在内存里**。"""

    image: "object"                       # np.ndarray，类型写成 object 以便无 numpy 环境也能导入本模块
    ts: float = field(default_factory=time.time)
    source: str = "unknown"
    index: int = 0

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.image.shape[:2]
        return int(w), int(h)

    def channel_stats(self) -> dict:
        """整帧的粗统计（给「画面是否全黑/过曝」这类判断用；不含任何可识别人脸的信息）。"""
        return {
            "mean": round(float(self.image.mean()), 2),
            "max": int(self.image.max()),
            "min": int(self.image.min()),
        }


# ══════════════════════════════════════════════════════════════════
# 帧源：真摄像头 / 假帧源（可注入）


class FrameSource:
    """帧源接口。`read()` 返回 None 表示这一帧没取到（别抛异常，真实摄像头会间歇失败）。"""

    name = "base"

    def open(self) -> None:                 # pragma: no cover - 接口
        pass

    def read(self) -> Optional[Frame]:      # pragma: no cover - 接口
        raise NotImplementedError

    def close(self) -> None:                # pragma: no cover - 接口
        pass

    def describe(self) -> str:
        return self.name

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class OpenCvFrameSource(FrameSource):
    """真摄像头（cv2.VideoCapture）。

    坑：真实摄像头**第一帧经常是空的**（驱动刚打开）——所以 open() 之后吃掉几帧预热；
        非 Windows 上没有摄像头时 VideoCapture 会静默返回空帧，所以 open() 必须显式判一次。
    """

    def __init__(self, index: int = 0, width: int = 0, height: int = 0, warmup: int = 3):
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.warmup = max(0, int(warmup))
        self.name = f"camera:{self.index}"
        self._cap = None
        self._count = 0

    def open(self) -> None:
        _require_vision()
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"打不开摄像头 #{self.index}。常见原因：① 被别的程序占用 ② 本机没有摄像头 "
                f"③ Windows 相机权限被关（设置 → 隐私和安全性 → 相机 → 允许应用访问相机）"
            )
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        for _ in range(self.warmup):        # 预热：丢弃驱动刚打开时的空帧
            cap.read()

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise CameraError("帧源未打开：先 open()（或用 with 语句）")
        ok, img = self._cap.read()
        if not ok or img is None:
            return None                     # 单帧失败是正常的，别把整条链路炸掉
        self._count += 1
        return Frame(image=img, source=self.name, index=self._count)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> str:
        if self._cap is None:
            return f"{self.name}（未打开）"
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cv2 else 0
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cv2 else 0
        return f"{self.name}（{w}x{h}）"


class FakeFrameSource(FrameSource):
    """假帧源：给测试和 CI 用（**不需要真摄像头**）。

    可以喂固定图、也可以生成确定性噪点图；用来验证「取帧 → 检测 → 观察结果」整条链路。
    """

    def __init__(self, frames: Optional[Sequence["object"]] = None, size: Tuple[int, int] = (320, 240),
                 count: int = 1, seed: int = 20261003, blank: bool = False, source: str = "fake"):
        self._given = list(frames) if frames else None
        self._size = size
        self._count = int(count)
        self._seed = int(seed)
        self._blank = bool(blank)
        self.name = source
        self.opened = False
        self._n = 0

    def open(self) -> None:
        self.opened = True

    def read(self) -> Optional[Frame]:
        self.opened = True
        if self._n >= self._count:
            return None
        w, h = self._size
        if self._given is not None:
            img = self._given[self._n % len(self._given)]
        else:
            _require_vision()
            if self._blank:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                rng = np.random.default_rng(self._seed + self._n)
                img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        self._n += 1
        return Frame(image=img, source=self.name, index=self._n)

    def close(self) -> None:
        self.opened = False

    def describe(self) -> str:
        return f"{self.name}（假帧源 · {self._count} 帧 · {self._size[0]}x{self._size[1]}）"


# ══════════════════════════════════════════════════════════════════
# 人脸检测（「有人在吗」——不含身份）


class FaceDetector:
    """检测器接口：`detect(image)` → 人脸框列表（按面积从大到小）。"""

    name = "base"

    def detect(self, image) -> List[Face]:      # pragma: no cover - 接口
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class HaarFaceDetector(FaceDetector):
    """Haar 级联正脸检测（CPU、离线、不需要下载模型）。

    精度低于深度学习模型，但**零依赖零下载**、在 CI 里也能真跑；Phase 2 要换成
    ONNX（如 YuNet / RetinaFace）时只需实现同一个 `FaceDetector` 接口。
    """

    name = "haar"

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: int = 30):
        _require_vision()
        if not haar_available():
            raise CameraError(
                "Haar 人脸检测不可用：需要 opencv **4.x**（5.x 已移除 Haar 级联与随包模型文件）。"
                "装回 4.x：pip install \"opencv-python-headless>=4.9,<5\""
            )
        path = cv2.data.haarcascades + _CASCADE_FILE
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise CameraError(f"Haar 级联文件加载失败：{path}（opencv 安装可能不完整）")
        self._cascade = cascade
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = int(min_size)

    def detect(self, image) -> List[Face]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)                       # 光照不均时更稳
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        faces = [Face(int(x), int(y), int(w), int(h), score=1.0) for x, y, w, h in rects]
        faces.sort(key=lambda f: f.area, reverse=True)      # 最大的排前面（后续「谁在说话」启发式要用）
        return faces

    def describe(self) -> str:
        return f"haar（scale={self.scale_factor} neighbors={self.min_neighbors}）"


class StubFaceDetector(FaceDetector):
    """替身检测器：固定返回给定的人脸框（测试用，确定性）。"""

    name = "stub"

    def __init__(self, faces: Optional[Sequence[Face]] = None):
        self.faces = list(faces or [])

    def detect(self, image) -> List[Face]:
        return list(self.faces)

    def describe(self) -> str:
        return f"stub（固定 {len(self.faces)} 张脸）"


# ══════════════════════════════════════════════════════════════════
# 观察结果：给 Agent 的「看一眼」


def annotate(frame: Frame, faces: Sequence[Face], color: Tuple[int, int, int] = (0, 200, 0)) -> "object":
    """在副本上画框（**不改原帧**）。返回新 ndarray，给 `look_image()` 落盘用。"""
    _require_vision()
    img = frame.image.copy()
    for i, f in enumerate(faces, 1):
        cv2.rectangle(img, (f.x, f.y), (f.x + f.w, f.y + f.h), color, 2)
        cv2.putText(img, f"#{i}", (f.x, max(12, f.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def encode_image_b64(image, fmt: str = "jpeg", quality: int = 80) -> str:
    """把画面编码成 base64（交给视觉模型看图用）。默认 JPEG —— 摄像头帧比截图大得多，JPEG 更省。"""
    _require_vision()
    if fmt.lower() in ("jpg", "jpeg"):
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise CameraError(f"图像编码失败（{fmt}）")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def observe(frame: Frame, faces: Sequence[Face], with_image: bool = False,
            fmt: str = "jpeg", quality: int = 80) -> dict:
    """把一帧 + 检测结果整理成结构化观察（**只有数字和可选的图，没有任何身份信息**）。"""
    w, h = frame.size
    return {
        "source": frame.source,
        "frame_index": frame.index,
        "size": [w, h],
        "faces": len(faces),
        "boxes": [f.as_dict() for f in faces],
        "channel_stats": frame.channel_stats(),
        "image_b64": (f"{fmt}:" + encode_image_b64(frame.image, fmt, quality)) if with_image else "",
    }


def format_observation(obs: dict) -> str:
    """观察 → 一句人话（给模型/日志看）。"""
    if obs["faces"] == 0:
        head = "画面里没有人脸"
    elif obs["faces"] == 1:
        head = "画面里有 1 张人脸"
    else:
        head = f"画面里有 {obs['faces']} 张人脸"
    boxes = "；".join(
        f"#{i + 1} 在 ({b['x']},{b['y']}) 大小 {b['w']}x{b['h']}"
        for i, b in enumerate(obs["boxes"])
    )
    tail = f"（画面 {obs['size'][0]}x{obs['size'][1]}，来自 {obs['source']}）"
    return f"{head}：{boxes}{tail}" if boxes else f"{head}{tail}"


def capture(source: FrameSource, detector: Optional[FaceDetector] = None,
            tries: int = 3) -> Tuple[Frame, List[Face]]:
    """取一帧并检测。真实摄像头单帧失败很正常 → 默认最多试 3 次。"""
    frame = None
    for _ in range(max(1, int(tries))):
        frame = source.read()
        if frame is not None:
            break
    if frame is None:
        raise CameraError("连续取帧失败（摄像头可能被占用或已拔出）")
    faces = detector.detect(frame.image) if detector is not None else []
    return frame, faces


def grab_frame(index: int = 0, detector: Optional[FaceDetector] = None, tries: int = 3):
    """开摄像头 → 取一帧 → 一定关掉。三个工具要的都是这一步，别各写一遍（开/关必须配对）。"""
    src = OpenCvFrameSource(index=int(index))
    try:
        src.open()
        return capture(src, _default_detector() if detector is None else detector, tries=tries)
    finally:
        src.close()


def probe_camera(index: int = 0) -> dict:
    """如实探测摄像头是否可用（不假装有）——给 `forge --doctor` 之类用。"""
    if np is None or cv2 is None:
        return {
            "ok": False,
            "reason": "missing_deps",
            "detail": "没装 opencv/numpy：pip install \"handcraft-agent[vision]\"",
        }
    src = OpenCvFrameSource(index=index, warmup=1)
    try:
        src.open()
        frame = src.read()
        if frame is None:
            return {"ok": False, "reason": "no_frame", "detail": "摄像头能打开但读不到帧（可能被占用）"}
        w, h = frame.size
        return {"ok": True, "reason": "ok", "detail": f"摄像头 #{index} 可用（{w}x{h}）", "size": [w, h]}
    except CameraError as e:
        return {"ok": False, "reason": "open_failed", "detail": str(e)}
    finally:
        src.close()


def haar_available() -> bool:
    """Haar 级联到底能不能用（**能力探测，不是只看 import**）。

    2026-10-03 实测：OpenCV **5.0 移除了 Haar** —— `cv2.CascadeClassifier` 和 `cv2.objdetect` 都没了，
    `cv2.data/haarcascades/` 也只剩个空目录（不再随包发模型文件）。所以必须探测
    「类在不在 + 模型文件在不在」，不能只判 `import cv2` 成功。
    """
    if np is None or cv2 is None:
        return False
    if not (hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")):
        return False
    try:
        return os.path.isfile(os.path.join(cv2.data.haarcascades, _CASCADE_FILE))
    except Exception:                        # pragma: no cover - 路径异常一律当不可用
        return False


def available() -> dict:
    """本机视觉能力盘点（如实汇报，不假装有）。"""
    ok = np is not None and cv2 is not None
    return {
        "numpy": np is not None,
        "opencv": cv2 is not None,
        "opencv_version": getattr(cv2, "__version__", "") if cv2 is not None else "",
        "haar": haar_available(),
        "detector": "haar" if haar_available() else "",
    }


# ══════════════════════════════════════════════════════════════════
# 工具：接进 Agent 的工具系统（A02 / A06 只读分级）

_last_detector: Optional[FaceDetector] = None


def _default_detector() -> FaceDetector:
    """默认检测器（Haar）。缓存一份 —— 级联加载一次就好。"""
    global _last_detector
    if _last_detector is None:
        _last_detector = HaarFaceDetector()
    return _last_detector


def register_camera_tools() -> bool:
    """把 `look` / `look_image` 注册进工具系统。

    **没装 opencv/numpy 就不挂** —— 工具列表里不出现，模型就不会规划它做不到的动作
    （和客户端执行器「未装则该能力如实报不可用」同一条原则）。
    """
    if np is None or cv2 is None:            # 没有视觉依赖：如实不挂
        return False
    from .tools import tool

    @tool(
        name="look",
        description=(
            "看一眼摄像头：返回画面尺寸、检测到的**人脸数量**和每张脸的框位置。"
            "只回数字，不含身份信息、不落盘。要拿图像本身（例如交给视觉模型看）用 look_image。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "摄像头编号，默认 0"},
                "with_image": {"type": "boolean", "description": "是否附带 base64 图像（默认 false）"},
            },
        },
        read_only=True,
    )
    def look(index: int = 0, with_image: bool = False) -> str:
        frame, faces = grab_frame(index)
        return format_observation(observe(frame, faces, with_image=bool(with_image)))

    @tool(
        name="look_image",
        description=(
            "看一眼摄像头并把**带框的画面存成文件**（默认 JPEG），返回文件路径。"
            "这是写操作（会在磁盘留图）——人脸属于敏感个人信息，只在确实需要时才用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "保存路径（.jpg/.png）"},
                "index": {"type": "integer", "description": "摄像头编号，默认 0"},
            },
            "required": ["path"],
        },
        read_only=False,                    # 落盘 = 有副作用，走 A06 写操作那一档
    )
    def look_image(path: str, index: int = 0) -> str:
        frame, faces = grab_frame(index)
        img = annotate(frame, faces)
        exp = os.path.expanduser(str(path))
        parent = os.path.dirname(os.path.abspath(exp))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not cv2.imwrite(exp, img):
            raise CameraError(f"写图失败：{exp}")
        return f"已保存 {exp}（{format_observation(observe(frame, faces))}）"

    return True
