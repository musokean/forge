"""在场感知（#18 Phase 3）—— 「谁在场、谁在说话」。

前三层的分工：

    camera.py   有没有人、脸在哪            （检测）
    faces.py    这是谁                     （登记 + 识别）
    presence.py 现在几个人、**谁在说话**    （跟踪 + 主动说话人）← 本模块

## 「谁在说话」怎么判（以及为什么这么判）

单麦克风**没有方向信息** ✗，Haar 检测器**不给关键点** ✗ —— 所以不做音源定位，也不做唇形识别。
用两个**可靠、零额外依赖**的信号叠加：

1. **嘴部区域的帧间运动** ✓ —— 把脸框下三分之一（嘴所在处）缩到很小（默认 32x16）再逐帧比，
   说话的人嘴在动，分数就高。用极小图 + 均值差，抗噪又便宜 ✓；
2. **脸区大小** ✓ —— 镜头前通常是谁离得近谁是主体（`camera` 已按面积降序，这里当平局时的次序）。

判定同样**不硬猜** ✓：最高分必须过 `min_motion`，**且**明显高于第二名（`ratio`），
否则回「不确定（没人在明显动嘴）」—— 比随便指一个强得多。

## 身份跟踪：一个人只该有一个名字

`Tracker` 按**质心距离**把每帧的检测框关联成**稳定的 track id**（带 TTL 与迟滞 ✓）：
同一个人小幅移动不会换 id；离场超过 TTL 才回收。名字**按 track 缓存**并**降频重认**
（`recognize_every` 帧一次）—— 既省算力，又不会因为某一帧识别失败就"人名的灯"乱闪 ✓。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:                          # 核心安装不含 numpy（只在 [vision] extra 里）
    import numpy as np
except ModuleNotFoundError:   # pragma: no cover
    np = None

__all__ = [
    "PresenceError",
    "Track",
    "Tracker",
    "MouthMotion",
    "PresenceMonitor",
    "mouth_region",
    "register_presence_tools",
]

# 判定阈值（可调）
DEFAULT_MIN_MOTION = 0.015    # 低于它 = 没人明显动嘴
DEFAULT_RATIO = 1.5           # 最高分至少是第二名的 1.5 倍，才算"就是他"
DEFAULT_TTL = 1.5             # 秒：超过这么久没跟上就认为离场
DEFAULT_MAX_DIST = 0.25       # 质心距离 / 脸宽 超过它就不认为是同一个人


class PresenceError(RuntimeError):
    """在场感知不可用（带了怎么办）。"""


def _require_numpy():
    if np is None:
        raise PresenceError("在场感知需要 numpy：pip install \"handcraft-agent[vision]\"")


# ══════════════════════════════════════════════════════════════════
# 嘴部区域与运动量


def mouth_region(image, box, size: Tuple[int, int] = (32, 16)):
    """取脸框**下三分之一**（嘴所在区域），缩到 `size`（宽 x 高）。

    为什么要缩小：说话时嘴部的像素级变化集中在很小一块，先降采样等于低通滤波 ——
    抗噪，且每帧运算量固定（与摄像头分辨率无关 ✓）。
    """
    _require_numpy()
    h, w = image.shape[:2]
    x, y, bw, bh = int(box.x), int(box.y), int(box.w), int(box.h)
    y0 = max(0, y + int(bh * 0.62))            # 下三分之一
    y1 = min(h, y + bh)
    x0, x1 = max(0, x), min(w, x + bw)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image[y0:y1, x0:x1]
    sw, sh = size
    # 最近邻降采样（自己算，避免依赖 cv2 的 resize；也让本模块能纯 numpy 测试 ✓）
    yi = (np.linspace(0, crop.shape[0] - 1, sh)).astype("int32")
    xi = (np.linspace(0, crop.shape[1] - 1, sw)).astype("int32")
    small = crop[yi][:, xi].astype("float32")
    return small


class MouthMotion:
    """嘴部运动评分：对每个 track 维护最近若干帧的「小图」，算帧间平均绝对差。

    分数含义：相对亮度变化的均值（0~1 量级）。说话时约 0.02~0.15，静止时接近 0（噪声级）。
    """

    def __init__(self, history: int = 6):
        self.history = max(2, int(history))
        self._regions: Dict[int, List["object"]] = {}
        self._scores: Dict[int, List[float]] = {}

    def update(self, track_id: int, small) -> float:
        """喂一帧该 track 的嘴部小图，返回**瞬时**运动分（与上一帧比）。

        帧差只在这里算一次，存进 `_scores`；`smooth()` 只对已经算好的分数做平滑（别再算一遍 ✗）。
        """
        if small is None:
            return 0.0
        regions = self._regions.setdefault(track_id, [])
        score = float(np.mean(np.abs(small - regions[-1])) / 255.0) if regions else 0.0
        regions.append(small)
        del regions[: max(0, len(regions) - self.history)]
        scores = self._scores.setdefault(track_id, [])
        scores.append(score)
        del scores[: max(0, len(scores) - self.history)]
        return score

    def smooth(self, track_id: int, alpha: float = 0.4) -> float:
        """对最近几帧的运动分做指数平滑（防止单帧抖动把「谁在说话」换来换去）。"""
        scores = self._scores.get(track_id) or []
        if not scores:
            return 0.0
        s = scores[0]
        for x in scores[1:]:
            s = (1 - alpha) * s + alpha * x
        return s

    def live_ids(self) -> List[int]:
        return list(self._regions)

    def forget(self, track_id: int) -> None:
        self._regions.pop(track_id, None)
        self._scores.pop(track_id, None)


# ══════════════════════════════════════════════════════════════════
# 跨帧跟踪


@dataclass
class Track:
    """一个人的一段连续在场记录。`track_id` 在一次访问内保持稳定。"""

    track_id: int
    box: "object"                                  # camera.Face
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    name: str = ""
    name_score: float = 0.0
    name_checked_frame: int = -10 ** 9
    mouth: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return float(self.box.x + self.box.w / 2), float(self.box.y + self.box.h / 2)

    def as_dict(self) -> dict:
        return {"track_id": self.track_id, "name": self.name or f"未登记#{self.track_id}",
                "score": round(self.name_score, 3), "mouth": round(self.mouth, 4),
                "misses": self.misses,          # 这一帧没跟上它的次数（遮挡/侧脸时有用）
                "box": [int(self.box.x), int(self.box.y), int(self.box.w), int(self.box.h)]}


class Tracker:
    """质心关联的简易多目标跟踪（够用且零依赖）。

    规则：每帧把检测框关联到「质心距离最近且不超过 `max_dist × 脸宽`」的已有 track；
    关联不上的开新 track；连续 `ttl` 秒没被关联的 track 视为离场并回收。
    """

    def __init__(self, ttl: float = DEFAULT_TTL, max_dist: float = DEFAULT_MAX_DIST):
        self.ttl = float(ttl)
        self.max_dist = float(max_dist)
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.frame_index = 0

    def update(self, faces: Sequence["object"], now: Optional[float] = None) -> List[Track]:
        now = time.time() if now is None else float(now)
        self.frame_index += 1
        unmatched = list(range(len(faces)))
        # 先给每个已有 track 找最近的框（贪心即可：人脸数很小）
        for tid, tr in list(self._tracks.items()):
            if not unmatched:
                tr.misses += 1
                continue
            cx, cy = tr.center
            best, best_d = None, None
            for i in unmatched:
                f = faces[i]
                fx, fy = f.x + f.w / 2, f.y + f.h / 2
                d = math.hypot(fx - cx, fy - cy) / max(1.0, float(f.w))
                if best_d is None or d < best_d:
                    best, best_d = i, d
            if best is not None and best_d is not None and best_d <= self.max_dist:
                f = faces[best]
                tr.box, tr.last_seen, tr.hits, tr.misses = f, now, tr.hits + 1, 0
                unmatched.remove(best)
            else:
                tr.misses += 1
        for i in unmatched:                     # 剩下的开新 track
            f = faces[i]
            self._tracks[self._next_id] = Track(self._next_id, f, now, now)
            self._next_id += 1
        for tid, tr in list(self._tracks.items()):   # 回收离场的
            if now - tr.last_seen > self.ttl:
                del self._tracks[tid]
        return sorted(self._tracks.values(), key=lambda t: t.box.area, reverse=True)

    def active(self) -> List[Track]:
        return sorted(self._tracks.values(), key=lambda t: t.box.area, reverse=True)


# ══════════════════════════════════════════════════════════════════
# 合起来：在场 + 谁在说话


class PresenceMonitor:
    """把「跟踪 + 嘴部运动 + 识别（降频）」合成一个观察器。

    `recognizer` 是回调：吃 (frame_image, box) → (名字, 相似度)；不传就只跟踪不定名。
    `recognize_every`：每 N 帧才重新认一次（**省算力**），中间沿用缓存的名字。
    """

    def __init__(self, recognizer: Optional[Callable] = None, recognize_every: int = 10,
                 min_motion: float = DEFAULT_MIN_MOTION, ratio: float = DEFAULT_RATIO,
                 ttl: float = DEFAULT_TTL, mouth_size: Tuple[int, int] = (32, 16)):
        _require_numpy()
        self.recognizer = recognizer
        self.recognize_every = max(1, int(recognize_every))
        self.min_motion = float(min_motion)
        self.ratio = float(ratio)
        self.tracker = Tracker(ttl=ttl)
        self.motion = MouthMotion()
        self.mouth_size = mouth_size
        self.frames = 0
        self.recognize_calls = 0          # 给测试断言「降频真的生效」用 ✓

    def observe(self, frame, faces: Sequence["object"], now: Optional[float] = None) -> List[Track]:
        """喂一帧（图 + 该帧检测到的人脸），返回更新后的 tracks（含名字与嘴动分）。

        `frame` 可以是 `camera.Frame`，也可以是**裸图像数组**（库使用者不必先包一层 ✓）。
        """
        image = getattr(frame, "image", frame)
        tracks = self.tracker.update(faces, now=now)
        self.frames += 1
        for tr in tracks:
            small = mouth_region(image, tr.box, self.mouth_size)
            self.motion.update(tr.track_id, small)
            tr.mouth = self.motion.smooth(tr.track_id)
            if self.recognizer is not None:
                # 新 track 的 name_checked_frame 是很小的负数 → 第一帧必然到期，会立刻认一次 ✓
                if (self.frames - tr.name_checked_frame) >= self.recognize_every:
                    self.recognize_calls += 1
                    tr.name_checked_frame = self.frames
                    name, score = self.recognizer(image, tr.box)
                    if name:                      # 认出名字就更新缓存；没认出**保留**上次的名字（不闪 ✓）
                        tr.name, tr.name_score = name, float(score)
        live = {t.track_id for t in tracks}
        for tid in [t for t in self.motion.live_ids() if t not in live]:
            self.motion.forget(tid)          # track 离场后清掉它的嘴动历史，别让 id 复用带上旧分
        return tracks

    def snapshot(self, tracks: Optional[Sequence[Track]] = None) -> dict:
        """给「谁在场、谁在说话」的结论 —— 拿不定就说拿不定 ✓。"""
        tracks = list(tracks if tracks is not None else self.tracker.active())
        if not tracks:
            return {"people": [], "speaking": None, "reason": "画面里没有人"}
        ranked = sorted(tracks, key=lambda t: t.mouth, reverse=True)
        best = ranked[0]
        second = ranked[1].mouth if len(ranked) > 1 else 0.0
        who = best.name or f"未登记#{best.track_id}"
        if best.mouth < self.min_motion:
            return {"people": [t.as_dict() for t in tracks], "speaking": None,
                    "reason": f"分数最高的 {who} 只有 {best.mouth:.4f}，低于阈值 {self.min_motion}"
                              f"（没人在明显动嘴）"}
        if len(ranked) > 1 and best.mouth < second * self.ratio:
            return {"people": [t.as_dict() for t in tracks], "speaking": None,
                    "reason": f"{who} 与另一个人的嘴动分太接近"
                              f"（{best.mouth:.4f} vs {second:.4f}），分不清谁在说"}
        return {"people": [t.as_dict() for t in tracks], "speaking": who,
                "score": round(best.mouth, 4), "reason": "嘴部运动最明显且明显领先"}


def format_presence(snap: dict, seconds: float = 0.0) -> str:
    """结论 → 一句人话。"""
    ppl = snap.get("people") or []
    if not ppl:
        return f"（观察 {seconds:.1f}s）画面里没有人。"
    names = "、".join(p["name"] for p in ppl)
    head = f"（观察 {seconds:.1f}s）在场 {len(ppl)} 人：{names}。"
    sp = snap.get("speaking")
    if sp:
        return head + f"正在说话：{sp}（嘴部运动 {snap.get('score', 0):.4f}）。"
    return head + f"谁在说话：不确定 —— {snap.get('reason', '')}。"


# ══════════════════════════════════════════════════════════════════
# 工具


def register_presence_tools() -> bool:
    """注册 `who_is_speaking`。没装 numpy/cv2 就不挂（不假装有）。"""
    if np is None:
        return False
    try:
        import cv2                                  # noqa: F401  只有真要开摄像头才需要
    except ModuleNotFoundError:                     # pragma: no cover
        return False
    from .tools import tool

    @tool(
        name="who_is_speaking",
        description=(
            "盯着摄像头看几秒，回答「现在有几个人、是谁、谁在说话」。"
            "靠**嘴部区域的帧间运动**判断（单麦没有方向信息），拿不准时会明确说「不确定」而不是乱指。"
            "名字来自人脸库（face_enroll 登记过的才有名字，其余显示为「未登记#N」）。"
        ),
        parameters={"type": "object", "properties": {
            "seconds": {"type": "number", "description": "观察时长秒数，默认 3"},
            "index": {"type": "integer", "description": "摄像头编号，默认 0"}}},
        read_only=True,
    )
    def who_is_speaking(seconds: float = 3.0, index: int = 0) -> str:
        from .camera import HaarFaceDetector, OpenCvFrameSource
        from .faces import FaceStore, crop_face, face_defaults, make_embedder
        dur = max(0.5, min(30.0, float(seconds)))
        det = HaarFaceDetector()
        store, emb = None, None
        try:
            store = FaceStore(face_defaults()["db"])
            emb = make_embedder()
        except Exception:
            store, emb = None, None      # 没登记过 / 没模型：照样能答「几个、谁在说话」，只是没名字 ✓

        def recognize(image, box):
            if store is None or emb is None:
                return "", 0.0
            m = store.match(emb.embed(crop_face(image, box)))
            return (m.name, m.score) if not m.unknown else ("", m.score)

        mon = PresenceMonitor(recognizer=recognize if store else None, recognize_every=5)
        src = OpenCvFrameSource(index=int(index))
        t0 = time.time()
        try:
            src.open()
            while time.time() - t0 < dur:
                frame = src.read()
                if frame is None:
                    continue
                mon.observe(frame, det.detect(frame.image))
        finally:
            src.close()
        return format_presence(mon.snapshot(), seconds=time.time() - t0)

    return True
