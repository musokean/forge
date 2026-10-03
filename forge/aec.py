"""回声消除（AEC）—— 外放也能「边听边说」，不用闭麦也不用按键（#11 语音 Phase 3 补完）。

**要解决的问题**：外放 + 麦克风会自激 —— 喇叭里的 TTS 被麦克风听回去，被当成「用户说话」，
于是它开始回答自己的话。半双工（播放期闭麦）与 PTT（按住空格）都能免耳机，但都牺牲了
「它说话时随时插话」。AEC 用**正在播的音频当参考信号**，自适应估计回声路径
（喇叭 → 空气 → 麦克风），从麦克风信号里减掉回声，于是：它说话时你插话，麦克风里只剩你。

**实现**：纯 numpy 的**分块 NLMS 自适应 FIR**（不引 C 扩展，CI 里可跑；本机实测 `pyaec`
在 Python 3.13 没有 wheel 装不上）。
  · 参考 x（播出去的音频）、麦克风 d（回声 + 近端说话）
  · ŷ = wᵀx 是估计回声，e = d − ŷ 是残差（近端语音 + 未消净的回声）
  · NLMS：w += μ·Σe·x_seg / (帧内平均功率 + ε)
  · **滤波器长度**要覆盖「输出缓冲延迟 + 声学延迟 + 混响」，所以不要求精确对齐时间戳，
    默认 200ms（16k = 3200 taps）——这是让 AEC 在真实链路里稳的关键。
  · **双讲检测（Geigel）**：近端一说话就冻结自适应，否则滤波器会被用户的声音带偏而发散。
  · 发散保护：残差比输入还大时也冻结。

**可插拔**（同 `STTEngine` / `TTSEngine`）：`make_aec("nlms" | "none" | "pyaec" | "speexdsp")`，
装了外部引擎就自动用（本机未装），没装就明确报错而不是静默降级。

**诚实边界**：NLMS 是**线性**回声消除，对「喇叭非线性失真 + 大音量」的消不干净；真实房间里
ERLE 通常 10-20dB，够把 VAD 从回声里救出来（默认阈值 0.012 vs 回声衰减后远低于它），
但不等同于 WebRTC 那一档（含非线性处理 + 舒适噪声）。参数需按房间实测调。
"""
from __future__ import annotations

import threading
import time
from collections import deque

try:                        # 核心安装不含 numpy（只在 [voice] extra 里）：模块仍要能导入，
    import numpy as np      # 真正用到 AEC 时再由 _require_numpy() 给出明确提示
except ModuleNotFoundError:  # pragma: no cover
    np = None

__all__ = ["AecEngine", "NullAec", "NlmsAec", "ReferenceTap", "make_aec", "available_engines"]


def _require_numpy():
    """AEC 需要 numpy；没装时给一句人话（而不是 ImportError 栈）。"""
    if np is None:                                           # pragma: no cover
        raise RuntimeError("回声消除需要 numpy：pip install numpy（或 pip install \"handcraft-agent[voice]\"）")


class AecEngine:
    """回声消除引擎接口：`process(mic, ref)` → 消回声后的麦克风信号。"""

    name = "base"
    samplerate = 16000

    def process(self, mic, ref):                      # pragma: no cover - 接口
        raise NotImplementedError

    def reset(self):
        pass

    def stats(self) -> dict:
        return {}


class NullAec(AecEngine):
    """不做任何处理（默认；也是「AEC 关掉」的显式表达）。"""

    name = "none"

    def process(self, mic, ref):
        return mic


class NlmsAec(AecEngine):
    """分块 NLMS 自适应 FIR 回声消除。

    · `frame_ms`   分块长度（与麦克风块一致最好，20ms 是常见值）
    · `filter_ms`  自适应滤波器长度 = 能覆盖的「延迟 + 混响」上限（越长越稳、越吃 CPU）
    · `mu`         步长（0.2-0.5 常用；太大发散、太小收敛慢）
    · `dtd_ratio`  Geigel 双讲门限：麦克风峰值 > 参考近期峰值 × 该比例 → 判为近端说话 → 冻结
    """

    name = "nlms"

    def __init__(self, samplerate: int = 16000, frame_ms: int = 20, filter_ms: int = 200,
                 mu: float = 0.35, eps: float = 1e-6, dtd_ratio: float = 2.0,
                 warmup_ms: int = 300):
        _require_numpy()
        self.samplerate = int(samplerate)
        self.frame = max(1, int(self.samplerate * frame_ms / 1000))
        self.taps = max(1, int(self.samplerate * filter_ms / 1000))
        self.mu = float(mu)
        self.eps = float(eps)
        self.dtd_ratio = float(dtd_ratio)
        # 冷启动收敛期：系数还在学房间响应时，残差里仍有回声 —— 这段时间不能把残差当「用户说话」
        self.warmup_frames = max(1, int(warmup_ms / max(1, frame_ms)))
        self.reset()

    # ---- 状态 ----
    def reset(self):
        self._w = np.zeros(self.taps, dtype=np.float32)        # 回声路径估计
        self._x = np.zeros(self.taps, dtype=np.float32)        # 参考信号延迟线
        self.frames = 0
        self.frozen_frames = 0
        self._echo_pow = 0.0                                   # 滚动：估计回声能量
        self._res_pow = 0.0                                    # 滚动：残差能量

    # ---- 主处理 ----
    def process(self, mic, ref):
        """消回声。`mic`/`ref` 同长即可（整段或单块都行），内部按 `frame` 分块保持延迟线连续。"""
        mic = np.asarray(mic, dtype=np.float32).reshape(-1)
        ref = np.asarray(ref, dtype=np.float32).reshape(-1)
        if mic.size == 0:
            return mic
        if ref.size < mic.size:                                # 参考短了就补零（宁可少消，不要错位）
            ref = np.concatenate([ref, np.zeros(mic.size - ref.size, dtype=np.float32)])
        out = np.empty(mic.size, dtype=np.float32)
        for i in range(0, mic.size, self.frame):
            n = min(self.frame, mic.size - i)
            out[i:i + n] = self._process_frame(mic[i:i + n], ref[i:i + n])
        return out

    def _process_frame(self, d, x):
        """单帧：估计回声 → 相减 → （未被冻结时）NLMS 更新。"""
        f = d.size
        xw = np.concatenate([self._x, x]).astype(np.float32)          # 长 taps + f
        # 每个样点的输入窗：Xrev[n] = [x[n], x[n-1], ..., x[n-taps+1]]
        # 注意切片：延迟线长 taps + f，帧内第 i 个样点的输入窗**结束于 taps+i** ⇒ 取 [1:f+1]。
        # 2026-10-03 踩坑：写成 [:f] 会整体错一格 → 残差≈输入、权重几乎不更新（小例子里 w 恒为 0，
        # 实测 ERLE 只有 0.2dB），且表面看「有在跑」，极难发现。
        win = np.lib.stride_tricks.sliding_window_view(xw, self.taps)[1:f + 1][:, ::-1]
        y = win @ self._w                                             # 估计回声
        e = d - y                                                     # 残差

        echo_pow = float(np.mean(y * y))
        res_pow = float(np.mean(e * e))
        mic_pow = float(np.mean(d * d))
        self._echo_pow = 0.9 * self._echo_pow + 0.1 * echo_pow
        self._res_pow = 0.9 * self._res_pow + 0.1 * res_pow

        # ① 双讲检测（DTD）：**初始收敛期不冻结**（否则学不到房间响应），
        #    收敛后用「回声能量比」判：麦克风能量 ≫ 滤波器估计的回声能量 → 近端在说话。
        #    2026-10-03 踩坑：先用的是 Geigel（|d| > 比例 × 参考近期峰值）—— 语音**包络低谷**时
        #    |d| 相对 40ms 峰值轻易超 2 倍 → 长期误冻结（frozen 涨到 94/460 帧），ERC 只有几个 dB。
        #    能量比判据在「包络低谷」下自洽（y 跟着回声一起小），只在真双讲时冻结。
        warm = self.frames < self.warmup_frames
        near_end = (not warm) and echo_pow > 1e-9 and mic_pow > self.dtd_ratio * echo_pow
        # ② 发散保护：消完比不消还响说明滤波器被带偏了。
        # 2026-10-03 踩坑：判据写成 `res > mic` 毫无余量 —— 未收敛时残差本来就 ≈ 输入，
        # 浮点噪声让它在多数帧都成立 → 自适应几乎每帧被冻结，权重长不起来（ERLE 0.2dB）。
        # 必须给余量（这里 2×，即「明显变差」才算发散）。
        diverging = mic_pow > 1e-9 and res_pow > 2.0 * mic_pow
        if near_end or diverging:
            self.frozen_frames += 1
        else:
            # 块 NLMS 归一化：分母 = **每样本输入窗功率的均值**（≈ taps × σ²）。
            # 2026-10-03 实测踩坑（三次才对）：多除一个帧长 f 就等价于每步小 f 倍（20ms 时 20 倍）
            # → 权重每帧只动万分之几，肉眼像「在跑」但 ERLE 只有 0.2dB、残差≈输入。
            # 正确形式：Δw = μ·Σ_n e[n]·x_n / mean_n(‖x_n‖²)
            p = float(np.mean(np.sum(win * win, axis=1))) + self.eps
            self._w += self.mu * (win.T @ e) / p
            self._w = np.clip(self._w, -10.0, 10.0)                   # 数值兜底

        self._x = xw[-self.taps:].copy()
        self.frames += 1
        return e

    # ---- 观测 ----
    def converged(self) -> bool:
        """滤波器是否已可信：跑够 `warmup_ms`，或已经明显在消回声（ERLE ≥ 10dB）。

        冷启动期（真实产品也一样）残差里还有回声，此时把残差当人声会误触发 —— 这条用于门控。
        """
        return self.frames >= self.warmup_frames or self.erle_db() >= 10.0

    def erle_db(self) -> float:
        """回声抑制量（dB，滚动估计）：估计回声能量 / 残差能量。近端说话时会偏乐观，仅供观测。"""
        if self._res_pow <= 1e-12 or self._echo_pow <= 1e-12:
            return 0.0
        return float(10.0 * np.log10(self._echo_pow / self._res_pow))

    def stats(self) -> dict:
        return {"engine": self.name, "frames": self.frames, "frozen": self.frozen_frames,
                "erle_db": round(self.erle_db(), 1), "taps": self.taps}


# ══════════════════════════════════════════════════════════════════
# 外部引擎（装了就用，没装明确报错）
# ══════════════════════════════════════════════════════════════════

class _PyAecAdapter(AecEngine):
    """`pyaec`（C 扩展，含非线性处理）的适配器。本机 3.13 无 wheel，装不上 —— 留作可选。"""

    name = "pyaec"

    def __init__(self, samplerate: int = 16000, frame_ms: int = 20, filter_ms: int = 200):
        import pyaec                                            # noqa: F401  未装则在此抛错
        self._aec = pyaec.Aec(samplerate, frame_ms, filter_ms // frame_ms)

    def process(self, mic, ref):
        return self._aec.cancel_echo(list(np.asarray(mic, dtype=np.float32).reshape(-1)),
                                     list(np.asarray(ref, dtype=np.float32).reshape(-1)))


_ENGINES = {"none": NullAec, "nlms": NlmsAec}


def available_engines() -> list:
    """可用的引擎名（外部引擎装了才算）。"""
    names = ["none", "nlms"]
    for mod, name in (("pyaec", "pyaec"), ("speexdsp", "speexdsp")):
        try:
            __import__(mod)
            names.append(name)
        except Exception:
            pass
    return names


def make_aec(name: str = "nlms", **kw) -> AecEngine:
    """按名字造引擎。装了外部引擎也照样可用 `name="pyaec"`（否则明确报错，不静默降级）。"""
    name = (name or "nlms").lower()
    if name in ("none", "off", "null"):
        return NullAec()
    if name == "nlms":
        return NlmsAec(**kw)
    if name == "pyaec":
        return _PyAecAdapter(**kw)
    raise ValueError(f"未知 AEC 引擎：{name}（可用：{', '.join(available_engines())}）")


class ReferenceTap:
    """记录「刚播出去的音频」，供 AEC 取参考信号。

    AEC 要减掉的是**我们自己播的声音**，所以必须拿到与麦克风同一时间轴上的播放样本。
    默认播放走 ffplay（外部进程）→ 拿不到样本 ⇒ AEC 模式要用进程内播放（`SoundDeviceSink`）。

    时间轴：`push(samples, t)` 表示这段样本从时刻 `t`（`time.monotonic()`）开始播；
    `segment(t, n, lead_s)` 取「时刻 `t - lead_s` 结束、往前 n 个样本」的参考段（不足补零）。
    真实链路的输入/输出缓冲延迟与声学延迟用 `lead_s` 粗对齐即可 —— AEC 的滤波器长度（默认 200ms）
    本身就覆盖了延迟不确定性。
    """

    def __init__(self, samplerate: int = 16000, capacity_s: float = 30.0):
        _require_numpy()
        self.samplerate = int(samplerate)
        self.cap = int(capacity_s * self.samplerate)
        self._chunks = deque()                    # (t_start, np.ndarray)
        self._total = 0
        self._cursor_t = None                     # 流式取参考的游标（见 stream()）
        self._lock = threading.Lock()

    def push(self, samples, t: float = None):
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        t = time.monotonic() if t is None else float(t)
        with self._lock:
            self._chunks.append((t, arr))
            self._total += arr.size
            while self._chunks and self._total - self._chunks[0][1].size > self.cap:
                self._total -= self._chunks.popleft()[1].size

    def segment(self, t: float, n: int, lead_s: float = 0.0) -> np.ndarray:
        """取参考段（长度 n，缺的地方补零）。"""
        out = np.zeros(int(n), dtype=np.float32)
        if n <= 0:
            return out
        end_t = float(t) - float(lead_s)
        start_t = end_t - n / float(self.samplerate)
        with self._lock:
            chunks = list(self._chunks)
        for t0, arr in reversed(chunks):
            t1 = t0 + arr.size / float(self.samplerate)
            if t1 <= start_t:
                break
            lo, hi = max(start_t, t0), min(end_t, t1)
            if hi <= lo:
                continue
            a = int(round((lo - t0) * self.samplerate))
            b = a + int(round((hi - lo) * self.samplerate))
            dst = int(round((lo - start_t) * self.samplerate))
            seg = arr[a:b]
            if seg.size:
                out[dst:dst + min(seg.size, out.size - dst)] = seg[:max(0, out.size - dst)]
        return out

    def stream(self, n: int, lead_s: float = 0.0, t: float = None) -> np.ndarray:
        """**流式取参考**：第一次用时间戳锚定，之后**按样本计数推进**。

        为什么不能每块都用墙钟：墙钟在块间有毫秒级抖动（读一次、算一次的时间差），
        会让自适应滤波器把回声路径学「糊」（实测：w[0]=0.235/w[1]=0.36 且远端一堆杂抽头，
        ERLE 只剩 2dB；改成样本计数后权重干净地落在 0.6、ERLE 30dB）。
        前提：播放与采集用同一块声卡（同一时钟）——这正是本机与绝大多数外放场景的情况；
        跨设备长时间跑需要周期性重锚（传 `t` 即可重设游标）。
        """
        with self._lock:
            if self._cursor_t is None:
                self._cursor_t = (time.monotonic() if t is None else float(t)) - float(lead_s)
            end_t = self._cursor_t
            self._cursor_t = end_t + n / float(self.samplerate)
        return self.segment(end_t, n, lead_s=0.0)

    def reset_cursor(self):
        """下次 `stream()` 重新用时间戳锚定（换设备/长时间漂移后调）。"""
        with self._lock:
            self._cursor_t = None

    def seconds(self) -> float:
        with self._lock:
            return self._total / float(self.samplerate)

    def clear(self):
        with self._lock:
            self._chunks.clear()
            self._total = 0
            self._cursor_t = None
