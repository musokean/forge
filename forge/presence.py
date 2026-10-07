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
import threading
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
# 低于它 = 没人**明显**动嘴。**这个默认是按真机实测定的，不是拍的** ✓✓ ——
#
# 2026-10-03 实测（本机内置摄像头、同一人、同一批参数）：
#     说话时 嘴动分 中位 0.0199（max 0.0224）
#     安静时 嘴动分 中位 0.0159（max 0.0223）   ← **两个分布重叠**
# → 结论：**纯视觉的嘴部帧间运动分，分不开「说话」与「安静」** ✗✓（不是门限问题：
#   抬到 0.025 会连说话时的 0.0199 一起漏掉 ✗）。主导帧差的是**检测框每帧抖动 1~3 像素**
#   （整块画面平移），不是嘴在动 ✓。
# → 所以这个默认值只声称「**明显动嘴**」✓，取 0.03（高于实测两个分布的 max ≈0.022 ✓，
#   保证不在安静时误报 ✓）。**要判定「谁在说话」，必须用音频 VAD 门控**
#   （"有人在说话"由 VAD 可靠给出 ✓，视觉只回答"是谁在动嘴" ✓）—— 见 docs/faces.md 的已知边界。
DEFAULT_MIN_MOTION = 0.03
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
                 ttl: float = DEFAULT_TTL, mouth_size: Tuple[int, int] = (32, 16),
                 audio: Optional[Callable] = None):
        """`audio`：可调用的「**此刻有人在说话吗**」门控（如 `voice.LiveSpeechGate` ✓，None = 纯视觉 ✓）。

        **为什么要它**：视觉的嘴动分**分不开说话与安静**（实测分布重叠 ✗）→ 音频负责「有没有人在说」✓，
        视觉只负责「是谁」✓。音频说「没人说话」时，本模块**直接回没人说话** ✓（不再拿噪声当说话 ✓）。
        """
        _require_numpy()
        self.audio = audio
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
        """给「谁在场、谁在说话」的结论 —— 拿不定就说拿不定 ✓。

        **有音频门控时**（`self.audio`）：音频说「没人说话」→ 直接回没人说话 ✓（最可靠的一半 ✓）；
        音频说「有人在说」→ 再用视觉挑动嘴最明显的那个给名字 ✓。
        **没音频时**：退回纯视觉 —— 这时**只说「谁动嘴最明显」，不说「在说话」** ✗✓。
        """
        tracks = list(tracks if tracks is not None else self.tracker.active())
        audio: Optional[bool] = None
        if self.audio is not None:
            try:
                audio = bool(self.audio())
            except Exception:                        # 门控坏了别把整件事弄挂 ✓
                audio = None
        mode = "audio_gated" if audio is not None else "motion_only"
        base = {"people": [t.as_dict() for t in tracks], "mode": mode, "audio": audio}
        if not tracks:
            return {**base, "speaking": None, "reason": "画面里没有人"}
        if audio is False:
            return {**base, "speaking": None,
                    "reason": "音频判定：此刻没人在说话（视觉嘴动分不作数 ✓）"}
        ranked = sorted(tracks, key=lambda t: t.mouth, reverse=True)
        best = ranked[0]
        second = ranked[1].mouth if len(ranked) > 1 else 0.0
        who = best.name or f"未登记#{best.track_id}"

        # ★ 音频门控模式下：**"有没有人在说话"已经由音频定了** ✓✓ —— 视觉只负责"是谁"，
        #   **绝不能用嘴动阈值去否决** ✗✓（2026-10-03 真机踩到：音频报"有人说话"的那 2 秒，
        #   因为嘴动分 0.0281 < min_motion 0.03，工具回了"没人说话" —— 门控装反了 ✗）。
        if audio:
            if len(tracks) == 1:
                return {**base, "speaking": tracks[0].name or f"未登记#{tracks[0].track_id}",
                        "score": round(tracks[0].mouth, 4),
                        "reason": f"音频判定有人在说话 ✓；画面里只有一个人，归给他"
                                  f"（嘴动 {tracks[0].mouth:.4f}）"}
            if best.mouth >= second * self.ratio and best.mouth > 0.0:
                return {**base, "speaking": who, "score": round(best.mouth, 4),
                        "reason": f"音频判定有人在说话 ✓ + 嘴部运动最明显且明显领先"
                                  f"（{best.mouth:.4f} vs {second:.4f}）"}
            return {**base, "speaking": None,
                    "reason": f"音频判定有人在说话 ✓，但画面有 {len(tracks)} 个人、"
                              f"嘴动分又分不出谁（{best.mouth:.4f} vs {second:.4f}）→ 不硬指"}

        # 纯视觉模式：**只能说"谁动嘴最明显"，不能说"在说话"** ✗✓
        if best.mouth < self.min_motion:
            return {**base, "speaking": None,
                    "reason": f"分数最高的 {who} 只有 {best.mouth:.4f}，低于阈值 {self.min_motion}"
                              f"（没人在明显动嘴）"}
        if len(ranked) > 1 and best.mouth < second * self.ratio:
            return {**base, "speaking": None,
                    "reason": f"{who} 与另一个人的嘴动分太接近"
                              f"（{best.mouth:.4f} vs {second:.4f}），分不清谁在说"}
        return {**base, "speaking": who, "score": round(best.mouth, 4),
                "reason": "嘴部运动最明显且明显领先"}


def format_presence(snap: dict, seconds: float = 0.0) -> str:
    """结论 → 一句人话。"""
    ppl = snap.get("people") or []
    if not ppl:
        return f"（观察 {seconds:.1f}s）画面里没有人。"
    names = "、".join(p["name"] for p in ppl)
    head = f"（观察 {seconds:.1f}s）在场 {len(ppl)} 人：{names}。"
    if snap.get("audio") is False:
        return head + "此刻没人在说话（音频判定 ✓）。"
    sp = snap.get("speaking")
    if sp:
        if snap.get("audio"):
            return head + (f"正在说话：{sp}（音频判定有人在说话 ✓ + 视觉定人，"
                           f"嘴部运动 {snap.get('score', 0):.4f}）。")
        return head + f"嘴动最明显的是 {sp}（未接音频，**不能确认在说话** ✗；嘴部运动 {snap.get('score', 0):.4f}）。"
    return head + f"谁在说话：不确定 —— {snap.get('reason', '')}。"


def format_ambient(snap: dict, seconds: float = 0.0) -> str:
    """紧凑版现场描述（**塞进每轮 prompt 用** ✓，比 `format_presence` 短得多）。

    与 `format_presence` 的分工：那个是给**用户看**的完整结论 ✓（含判断依据与不确定说明 ✓）；
    这个是给**模型看**的环境事实 ✓ —— 只说「画面里有谁、叫什么、多像」✓，
    说不出的一律不说 ✗（没登记就只报「未登记#N」，没识别到就不报 ✓）。
    """
    ppl = snap.get("people") or []
    if not ppl:
        return "画面里没有人。"
    parts = []
    for p in ppl:
        nm = p.get("name") or ("未登记#%s" % p.get("track_id", "?"))
        sc = p.get("score")
        parts.append(nm + ("（%.2f）" % float(sc) if isinstance(sc, (int, float)) else ""))
    return "在场 %d 人：%s。" % (len(ppl), "、".join(parts))



class AvSpeaker:
    """用「音频包络 × 每人嘴动」的**互相关**判断**谁在说话** ✓✓（两人场景的关键缺口 ✓）。

    为什么需要：视觉嘴动分**分不开说话与安静** ✗（实测分布重叠），音频门控只给「有没有人说」✓
    —— 两个都答不了「**哪一个**人在说」✗。但**同步性**可以：说话的人嘴在动、同时有声音 ✓；
    没说话的人嘴动要么无关、要么不同步 ✗。所以对每个 track 算它与音频包络的**皮尔逊相关** ✓，
    最高的那位就是说话人 ✓。

    **纯 Python、不依赖 numpy** ✓（能在无 numpy 的环境里被测试 ✓）；数据靠**时间戳**对齐 ✓
    （音频零阶保持采样到每个嘴动样本的时刻 ✓）。

    **判不准就说不准** ✓✓（与项目一贯做法一致 ✓）：样本太少 → 说不清 ✓；没人跟声音同步 → 说不清 ✓；
    两个人相关都高且接近 → **说不清** ✓（这正是「两个人同时在说」的情形 ✓）。

    ⚠️ **采样率前提** ✗✓：嘴动分必须来自**足够快的连续帧**（≥5~10 Hz ✓）。探针默认 ~1 Hz 只能当"谁在场" ✓，
    做互相关需要「有人说话时连开摄像头」的突发采样 ✓（约 10 Hz ✓）—— 否则 1 秒粒度的"嘴动"是模糊量 ✗。
    """

    def __init__(self, window_s: float = 2.5, min_samples: int = 6, max_lag_s: float = 0.3,
                 min_corr: float = 0.35, margin: float = 0.15, max_audio_age_s: float = 0.35):
        self.window_s = float(window_s)
        self.min_samples = max(3, int(min_samples))
        self.min_corr = float(min_corr)
        self.margin = float(margin)
        self.max_audio_age_s = float(max_audio_age_s)
        self.max_lag_s = float(max_lag_s)                      # 「嘴唇领先声音」的搜索范围 ✓
        self._audio = []                      # [(t, rms)] ✓
        self._mouth = {}                      # track_id -> [(t, score)] ✓

    def on_audio(self, t: float, rms: float) -> None:
        """喂一个音频块的**能量包络** ✓（不是布尔 ✓ —— 布尔做不了相关 ✗）。"""
        self._audio.append((float(t), float(rms)))
        self._prune(float(t))

    def on_mouth(self, track_id, t: float, score: float) -> None:
        """喂某个 track 的**这一帧**嘴动分 ✓。"""
        self._mouth.setdefault(track_id, []).append((float(t), float(score)))
        self._prune(float(t))

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        self._audio = [(t, v) for t, v in self._audio if t >= cutoff]
        for tid in list(self._mouth):
            keep = [(t, v) for t, v in self._mouth[tid] if t >= cutoff]
            if keep:
                self._mouth[tid] = keep
            else:
                del self._mouth[tid]

    def _audio_at(self, t: float) -> Optional[float]:
        """嘴动样本时刻的音频包络 ✓（零阶保持；太久没有音频就当 0 ✓）。"""
        best = None
        for at, v in self._audio:
            if at <= t + 1e-9:
                best = (at, v)
            else:
                break
        if best is None:
            return None
        if t - best[0] > self.max_audio_age_s:
            return None
        return best[1]

    @staticmethod
    def _pearson(xs: List[float], ys: List[float]) -> float:
        """皮尔逊相关 ✓（任一侧方差为 0 → 0.0 ✓，不做除零 ✗）。"""
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx <= 1e-12 or syy <= 1e-12:
            return 0.0
        return float(sxy / (sxx ** 0.5 * syy ** 0.5))

    def pairs(self, track_id, lag_s: float = 0.0) -> List[Tuple[float, float, float]]:
        """该 track 的 **(时刻, 嘴动分, 音频包络)** ✓ —— 音频按 `lag_s` 平移 ✓，只留有音频的样本 ✓。

        `lag_s > 0` 表示「音频取更早的」= 允许**嘴唇领先声音** ✓✓（这是真实的 ✓：实测说话时唇动
        比声音早 100~200ms 量级 ✓）→ 所以调用方要在若干 lag 上取最大相关 ✓，而不是只看 lag=0 ✗。
        """
        out = []
        for t, score in self._mouth.get(track_id, []):
            a = self._audio_at(t - lag_s)
            if a is not None:
                out.append((t, score, a))
        return out

    def rank(self, max_lag_s: Optional[float] = None) -> List[dict]:
        """每人结果（按**最佳延迟**上的相关降序 ✓）；样本不足的也算出来但会标出来 ✓。

        `max_lag_s`：在 ±这个范围内搜最佳延迟 ✓（默认 0.3s ✓ —— 覆盖「嘴唇领先声音」✓）。
        """
        rows = []
        for tid in self._mouth:
            best_r, best_lag, best_n = -2.0, 0.0, 0
            for lag in self._lags(max_lag_s):
                pr = self.pairs(tid, lag_s=lag)
                if len(pr) < 2:
                    continue
                r = self._pearson([a for _t, _s, a in pr], [s for _t, s, _a in pr])
                if r > best_r:
                    best_r, best_lag, best_n = r, lag, len(pr)
            if best_n == 0:                      # 一个可用样本都没有 ✓
                pr = self.pairs(tid)
                best_r, best_n = 0.0, len(pr)
            rows.append({"track_id": tid, "n": best_n, "corr": max(0.0, best_r), "lag": best_lag})
        rows.sort(key=lambda r: (-r["corr"], r["n"]))
        return rows

    def _lags(self, max_lag_s: Optional[float]) -> List[float]:
        """待搜的延迟列表 ✓（步长取采样间隔的量级 ✓）。"""
        m = self.max_lag_s if max_lag_s is None else float(max_lag_s)
        if m <= 0:
            return [0.0]
        steps = 9
        return [round(-m + 2 * m * i / (steps - 1), 4) for i in range(steps)]

    def decide(self) -> dict:
        """谁在说话 ✓ —— **判不准就返回 None + 原因** ✓✓。"""
        rows = self.rank()
        if not rows:
            return {"track_id": None, "corr": 0.0, "margin": 0.0, "reason": "没有画面里的人"}
        enough = [r for r in rows if r["n"] >= self.min_samples]
        if not enough:
            return {"track_id": None, "corr": rows[0]["corr"], "margin": 0.0,
                    "reason": "样本不足（需要 %d 个同步样本，现在最多 %d）" % (self.min_samples, rows[0]["n"])}
        best = enough[0]
        if best["corr"] < self.min_corr:
            return {"track_id": None, "corr": best["corr"], "margin": 0.0,
                    "reason": "没人跟声音明显同步（最高相关系数 %.2f < %.2f）" % (best["corr"], self.min_corr)}
        second = enough[1]["corr"] if len(enough) > 1 else -1.0
        gap = best["corr"] - second
        if len(enough) > 1 and gap < self.margin:
            return {"track_id": None, "corr": best["corr"], "margin": gap,
                    "reason": "两个人都在动、相关也接近（%.2f vs %.2f）→ 分不出" % (best["corr"], second)}
        return {"track_id": best["track_id"], "corr": best["corr"],
                "margin": gap if len(enough) > 1 else 1.0, "lag": best.get("lag", 0.0),
                "reason": "与声音同步（r=%.2f，延迟 %.2fs）" % (best["corr"], best.get("lag", 0.0))}



class SpeechBurst:
    """「**有人说话时才连开摄像头**」的突发采样 ✓✓ —— 给「音频 × 嘴动」互相关喂数据。

    **为什么必须这样** ✗✓：探针默认 ~1Hz ✓，而 `grab_frame` 每帧开关摄像头 ≈1fps ✗ → 这样的
    「嘴动」是 1 秒粒度的模糊量 ✗，根本做不了互相关 ✗。所以：声音一起来 → **连开**摄像头跑 ~10fps ✓
    （一次几秒 ✓），声音停了再关 ✓。

    **隐私** ✓：摄像头**只在检测到人声时开** ✓✓（比一直开着克制得多 ✓）；仍不开图、不存图 ✓。

    **可测试** ✓✓：时钟、音频门控、帧源、检测器**全部可注入** ✓ → `step()` 能被测试**完全驱动** ✓
    （不需要麦克风、不需要摄像头、不需要真人 ✓✓ —— 与项目既有做法一致 ✓）。
    """

    def __init__(self, monitor, gate, av: Optional["AvSpeaker"] = None, source=None, detector=None,
                 *, fps: float = 10.0, stop_silence_s: float = 0.6, max_burst_s: float = 20.0,
                 min_loud_ms: int = 200, clock=None, paused=None, verbose: bool = False):
        self.monitor = monitor
        self.gate = gate
        self.av = av if av is not None else AvSpeaker()
        self.source = source
        self.detector = detector
        self.fps = max(1.0, float(fps))
        self.stop_silence_s = float(stop_silence_s)
        self.max_burst_s = float(max_burst_s)
        self.min_loud_ms = int(min_loud_ms)
        self.clock = clock or time.monotonic
        # `paused`：**它自己在说话时**要暂停突发采样 ✓✓ —— 半双工只闭了主循环的麦 ✗，
        #   突发采样用的是**独立**的音频流 ✓ → 不暂停就会把 forge 自己的声音当成"有人在说" ✗✓
        #   （于是采样期间嘴动与声音根本不同步 → 判定必错 ✗）。接线时传 `PlaybackHandle.playing` ✓。
        self.paused = paused
        self.verbose = bool(verbose)      # 验收时把判定打出来 ✓（数字才是证据 ✓）
        self._bursting = False
        self._t0 = 0.0
        self._quiet_since: Optional[float] = None
        self._last_frame_t = -1.0
        self._last_audio_mono = 0.0
        self.bursts = 0
        self.frames = 0
        self.last_error = ""                 # 取帧失败**只报一次** ✗✓（别静默 ✗）
        self.last_result: Optional[dict] = None   # 最近一轮「谁在说」的判定 ✓（喂对话用 ✓）
        self._hold = None

    # ── 相机开关（带着锁 ✓：与工具串行，别抢崩 ✓）────────────────
    def _open(self) -> None:
        if self.source is None:              # 自建相机：**要独占** ✓（工具会等，而不是崩 ✗✓）
            from .camera import OpenCvFrameSource, hold_camera
            self._hold = hold_camera()
            self._hold.__enter__()
            self.source = OpenCvFrameSource(index=0)
        self.source.open()                   # 注入进来的源也要 open ✓（测试就靠这条 ✓）
        if self.detector is None:
            from .camera import make_detector
            self.detector = make_detector()

    def _close(self) -> None:
        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                pass
            self.source = None
        if self._hold is not None:
            try:
                self._hold.__exit__(None, None, None)
            except Exception:
                pass
            self._hold = None

    @property
    def bursting(self) -> bool:
        return self._bursting

    def step(self) -> Optional[dict]:
        """推进一次：`None` = 继续；返回 dict = 这一轮突发结束，里面是**谁在说**的判定 ✓。"""
        now = self.clock()
        if self._is_paused():
            if self._bursting:                   # 它开始说话了 → 立刻收尾并放掉相机 ✓
                self._close()
                self._bursting = False
            return None
        for mono, rms in self._new_audio():
            self.av.on_audio(mono, rms)
        try:
            speaking = bool(self.gate.speaking())
        except Exception:
            speaking = False
        if not self._bursting:
            if speaking and self.gate.loud_ms() >= self.min_loud_ms:
                self._open()
                self._bursting = True
                self._t0 = now
                self._quiet_since = None
                self._last_frame_t = -1.0
                self.bursts += 1
            return None
        if now - self._last_frame_t >= 1.0 / self.fps:
            self._last_frame_t = now
            self._read_one(now)
        if speaking:
            self._quiet_since = None
        elif self._quiet_since is None:
            self._quiet_since = now
        over = self._quiet_since is not None and (now - self._quiet_since) >= self.stop_silence_s
        if over or (now - self._t0) >= self.max_burst_s:
            self._close()
            self._bursting = False
            out = self.av.decide()
            names = {}
            try:
                for tr in self.monitor.tracker.active():
                    names[tr.track_id] = getattr(tr, "name", "") or ""
            except Exception:
                pass
            out["names"] = names
            out["frames"] = self.frames
            self.last_result = out               # ← 记住，给注入用 ✓
            if self.verbose:
                _nm = (out.get("names") or {}).get(out.get("track_id")) if out.get("track_id") is not None else None
                print("[谁在说] %s（%.1fs、%d 帧）" % (
                    ("%s · r=%.2f · 延迟%.2fs" % (_nm or ("#%s" % out["track_id"]), out.get("corr", 0.0), out.get("lag", 0.0))
                     if out.get("track_id") is not None else "分不出：" + str(out.get("reason"))),
                    (self.clock() - self._t0), out.get("frames", 0)), flush=True)
            return out
        return None

    def _is_paused(self) -> bool:
        """它在说话吗（是就别采样 ✓）。钩子坏了当作"没在说" ✓，别把整件事弄挂 ✓。"""
        if self.paused is None:
            return False
        try:
            return bool(self.paused())
        except Exception:
            return False

    def _new_audio(self) -> list:
        """只取**还没喂过**的音频样本 ✓（避免同一条样本反复进相关 ✓）。"""
        try:
            series = self.gate.energy_series(since_mono=self._last_audio_mono)
        except Exception:
            return []
        fresh = [(m, r) for m, r in series if m > self._last_audio_mono]
        if fresh:
            self._last_audio_mono = max(m for m, _r in fresh)
        return fresh

    def _read_one(self, now: float) -> None:
        frame = None
        try:
            frame = self.source.read()
        except Exception as exc:                 # 取帧失败**要说话** ✗✓（今天栽过 ✗）
            if not self.last_error:
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                print("[现场身份] 突发采样取帧失败（只报一次）：%s" % self.last_error)
            return
        if frame is None:
            return
        image = getattr(frame, "image", frame)
        try:
            faces = self.detector.detect(image)
            tracks = self.monitor.observe(frame, faces, now=now)
        except Exception as exc:
            if not self.last_error:
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                print("[现场身份] 突发采样处理失败（只报一次）：%s" % self.last_error)
            return
        self.frames += 1
        for tr in tracks:
            self.av.on_mouth(tr.track_id, now, float(getattr(tr, "mouth", 0.0) or 0.0))

    # ── 线程（真机用 ✓；测试直接调 step() ✓）─────────────────────
    def start(self) -> "SpeechBurst":
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="speech-burst", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:                     # pragma: no cover - 需要真麦克风
        while not getattr(self, "_stop", threading.Event()).is_set():
            try:
                self.step()
            except Exception:
                pass
            time.sleep(1.0 / self.fps)

    def close(self) -> None:
        try:
            if getattr(self, "_stop", None) is not None:
                self._stop.set()
            if getattr(self, "_thread", None) is not None:
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        self._close()


class PresenceProbe:
    """后台按 ~1Hz 看一眼「画面里有谁、刚才谁在说话」，产出**一行可注入对话的现场描述** ✓。

    为什么单独一个类：语音轮的主循环是**阻塞式**的（等转写、等模型、等播放 ✓），没法在里面
    同步看摄像头 ✗ —— 于是用轻量后台线程定期看一眼，**每次现开现关**摄像头 ✓（不长期占设备 ✓）。

    隐私：只读取**特征与相似度**，不保存任何图像 ✓；**默认不启动**，调用方显式 `start()` ✓
    （语音轮里要 `--identify` 才开 ✓）。读不到脸 / 没模型 / 库是空的，都只影响「有没有名字」✓，
    绝不假装有身份 ✗。
    """

    def __init__(self, interval_s: float = 1.0, index: int = 0, audio=None,
                 store=None, embedder=None, detector=None, ttl_s: float = 2.5, look_s: float = 1.0,
                 av_result=None):
        _require_numpy()
        self.interval_s = max(0.2, float(interval_s))
        self.index = int(index)
        self.look_s = float(look_s)
        self.ticks = 0
        self._ambient = ""
        self._err = ""
        self.first_error = ""       # 第一次失败的**真实原因**（只报一次 ✓，别吞 ✗）
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._detector = detector
        self._store = store
        self._embedder = embedder
        self._av_result = av_result      # 取「谁在说」的 AV 判定（可选 ✓）：多人时比嘴动更可靠 ✓
        # 跟踪器 TTL 必须**大于轮询间隔** ✓，否则每次轮询都被当成新目标（1Hz 下 ttl<1s 就废 ✗）
        self._mon = PresenceMonitor(recognizer=None, recognize_every=1,
                                    ttl=max(ttl_s, self.interval_s * 2.5), audio=audio)
        if self._store is not None and self._embedder is not None:
            self._mon.recognizer = self._make_recognizer()

    def _make_recognizer(self):
        """用**和登记完全相同**的配方（对齐路径 ✓），避免两条管线混用 ✗。

        2026-10-06 真机踩到 ✗✓：原先这里自己拼 `embedder.embed(crop_face(...))`（裁剪 ✗），而登记走的是
        `embed_face` 的**对齐**路径 ✓ —— 同一张脸同一帧，两条路的向量余弦只有 **0.537** ✗，匹配分从 **0.762
        掉到 0.448**（贴着 0.36 阈值 ✓ → 稍一变姿态就报「未登记」✗，真机就是这个症状 ✓）。
        **别混用两条管线** ✗✓：早前那句「对齐 vs 裁剪没差别」测的是各自**内部**的匹配，从没测过**混用** ✗。
        """
        from .faces import crop_face

        cache = {"img": None, "rows": None}

        def _rows(image):
            """这一帧的人脸行（按需算一次 ✓，避免每个目标都重跑检测 ✗）。"""
            det = self._detector
            if det is None or not hasattr(det, "detect_with_landmarks"):
                return None
            if cache["img"] is image:
                return cache["rows"]
            try:
                found = det.detect_with_landmarks(image)
            except Exception:
                found = None
            cache["img"], cache["rows"] = image, (found or None)
            return cache["rows"]

        def _box4(b):
            """取框的四个数 ✓ —— `Face`（dataclass，有 .x/.y/.w/.h ✓）与 `(x,y,w,h)` 两种形态都吃 ✓。"""
            if hasattr(b, "x"):
                return (b.x, b.y, b.w, b.h)
            return (b[0], b[1], b[2], b[3])

        def _row_for(rows, box):
            """按框位置找对应的人脸行 ✓（同一 detector、同一帧 → 顺序一致 ✓）。"""
            bx = _box4(box)
            best, best_i = None, 0
            for i, item in enumerate(rows):
                f = _box4(item[0])            # item = (Face, row) ✓ —— 不是 (框, row) ✗✓
                d = sum(abs(f[k] - bx[k]) for k in range(4))
                if best is None or d < best:
                    best, best_i = d, i
            return rows[best_i][1]

        def recognize(image, box):
            try:
                rows = _rows(image)
                if rows and hasattr(self._embedder, "embed_aligned"):
                    vec = self._embedder.embed_aligned(image, _row_for(rows, box))   # 与登记同路 ✓
                else:
                    vec = self._embedder.embed(crop_face(image, box))                # 退路（无 YuNet 时 ✓）
                m = self._store.match(vec)
                return (m.name, m.score) if not m.unknown else ("", m.score)
            except Exception as exc:                # 静默吞掉 ⇒ 现场全是「未登记」而无人知道为什么 ✗✓
                if not self.first_error:
                    self.first_error = "recognize: %s: %s" % (type(exc).__name__, exc)
                    print("[现场身份] 识别失败（只报一次，不影响语音 ✓）：%s" % self.first_error)
                return "", 0.0
        return recognize

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:                    # 探测失败**绝不能**拖垮语音轮 ✓
                with self._lock:
                    self._err = "%s: %s" % (type(exc).__name__, exc)
            wait = self.interval_s - (time.monotonic() - t0)
            if wait > 0:
                self._stop.wait(wait)

    def _tick(self) -> None:
        from .camera import grab_frame, make_detector
        if self._detector is None:
            self._detector = make_detector()
        # 工具（face_who/look）可能与探针抢摄像头 ✗ → 先礼让重试 ✓；仍失败就跳过本次，
        # 保留上一次观察（**别**把现场信息清空 ✗），并记下原因 ✓
        frame = faces = None
        last = None
        for attempt in range(3):
            try:
                frame, faces = grab_frame(index=self.index, detector=self._detector)
                break
            except Exception as exc:
                last = exc
                time.sleep(0.05 * (attempt + 1))
        if frame is None:
            raise last if last is not None else RuntimeError("摄像头取帧失败")
        self._mon.observe(frame, faces)
        text = format_ambient(self._mon.snapshot(), seconds=self.look_s)
        self.ticks += 1
        with self._lock:
            self._ambient = text
            self._err = ""

    def ambient_for_answer(self) -> str:
        """回答前用的现场信息：**在场有谁** + （画面里只有一个人时）**刚才说话的就是他** ✓。

        多人时视觉嘴动分**分不出谁在说** ✗（实测分布重叠 ✓）→ 如实说「无法确定」✓，不硬指 ✗。
        """
        snap = self._mon.snapshot()
        base = format_ambient(snap)
        ppl = snap.get("people") or []
        av = self._av_line()
        if av:
            return base + av
        if len(ppl) == 1:
            nm = ppl[0].get("name") or "画面里那位（未登记）"
            return base + "刚才说话的是 %s（画面里只有一人 ✓）。" % nm
        if len(ppl) > 1:
            return base + "刚才有人说话，但画面里有 %d 人、嘴动分分不出是谁 ✗。" % len(ppl)
        return base + "刚才说话的人不在画面里。"

    def _av_line(self) -> str:
        """把「音频 × 嘴动」的判定翻成一句注入用的话 ✓✓ —— **判不准就直说** ✗✓。

        比「嘴动最明显」（1Hz 探针只能给这个 ✗）可靠得多：它比的是**与声音的同步** ✓，
        两人场景下这是唯一能指认的依据 ✓；两人同时在说 → 这里会明确回「分不出」✓✓。
        """
        get = getattr(self, "_av_result", None)      # 半构造对象（测试用 __new__）也不能炸 ✓
        if get is None:
            return ""
        try:
            r = get() or {}
        except Exception:
            return ""
        if not r:
            return ""
        tid = r.get("track_id")
        names = r.get("names") or {}
        if tid is None:
            return "刚才有人说话，但%s。" % (r.get("reason") or "分不出是谁")
        nm = names.get(tid) or ("未登记#%s" % tid)
        return "刚才说话的是 %s（与声音同步 r=%.2f）。" % (nm, float(r.get("corr") or 0.0))

    @property
    def ambient(self) -> str:
        """最近一次的现场描述（可直接喂给 `Agent.set_ambient` ✓）。"""
        with self._lock:
            return self._ambient

    @property
    def error(self) -> str:
        with self._lock:
            return self._err


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
            "⚠️ **它不能判定「在说话」** ✗：本工具靠嘴部区域的帧间运动判断，而真机实测证明"
            "说话（中位 0.0199）与安静（中位 0.0159）两个分布重叠、没有门限能分开 "
            "（检测框每帧抖 1~3 像素就盖过嘴动贡献）。所以它只回答「谁动嘴更明显」✓，"
            "拿不准时明确说「不确定」；要判定谁在说话需配合音频 VAD（见 docs/faces.md 的已知边界）。"
            "名字来自人脸库（face_enroll 登记过的才有名字，其余显示为「未登记#N」）。"
        ),
        parameters={"type": "object", "properties": {
            "seconds": {"type": "number", "description": "观察时长秒数，默认 3"},
            "index": {"type": "integer", "description": "摄像头编号，默认 0"}}},
        read_only=True,
    )
    def who_is_speaking(seconds: float = 3.0, index: int = 0) -> str:
        from .camera import OpenCvFrameSource, make_detector
        from .faces import FaceStore, crop_face, face_defaults, make_embedder
        from .voice import open_live_speech_gate
        dur = max(0.5, min(30.0, float(seconds)))
        det = make_detector()                        # 有 YuNet 就用 YuNet ✓
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

        # 音频门控：有麦克风就用它判「此刻有没有人在说话」✓（声音负责这一半最可靠 ✓）；
        # 拿不到麦克风 → None → 退回纯视觉，**那时只报「谁动嘴最明显」，不声称在说话** ✗✓
        gate = open_live_speech_gate()
        mon = PresenceMonitor(recognizer=recognize if store else None, recognize_every=5, audio=gate)
        src = OpenCvFrameSource(index=int(index))
        t0 = time.time()
        try:
            src.open()
            while time.time() - t0 < dur:
                frame = src.read()
                if frame is None:
                    continue
                mon.observe(frame, det.detect(frame.image))
            snap = mon.snapshot()
        finally:
            src.close()
            if gate is not None:
                gate.close()
        return format_presence(snap, seconds=time.time() - t0)

    return True
