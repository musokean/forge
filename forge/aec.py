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

    def __init__(self, samplerate: int = 16000, frame_ms: int = 20, filter_ms: int = 100,
                 mu: float = 0.35, eps: float = 1e-6, dtd_ratio: float = 2.0,
                 warmup_ms: int = 300, w_max: float = 1.5, bypass_ratio: float = 1.25):
        _require_numpy()
        self.samplerate = int(samplerate)
        self.frame = max(1, int(self.samplerate * frame_ms / 1000))
        self.taps = max(1, int(self.samplerate * filter_ms / 1000))
        self.mu = float(mu)
        self.eps = float(eps)
        self.dtd_ratio = float(dtd_ratio)
        self.w_max = float(w_max)                # ‖w‖ 上限（真实房间回声路径增益不可能到 1 量级）
        self.bypass_ratio = float(bypass_ratio)  # 残差比输入还响 → 这一帧旁路（AEC 不该消得更差）
        self.bypass_frames = 0                   # 旁路次数（可观测：判断参考是否靠谱）
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
        self._mic_pow = 0.0                                    # 滚动：麦克风能量（旁路判据用平滑值）

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
        self._mic_pow = 0.9 * self._mic_pow + 0.1 * mic_pow


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
        # ⓪ 硬保险：**旁路**（这一帧原样输出输入、不消）—— 只在「明显更差」时触发。
        #    AEC 不该把链路消得更差：滤波器学坏时（参考错段/延迟超出跨度/房间突变/设备换位）兜住。
        #    ★ 2026-10-03 真机端到端复现踩到的坑：第一版写成「**每帧**无条件不超输入（无余量）」→
        #    未收敛时残差本来就常比输入略响 → **54% 的帧被旁路 → AEC 等于被关掉** → 它听回自己的话
        #    （用户实测原话：「他把自己说的也当作我的输入了」）。判据因此必须：① 用**平滑**能量
        #    （单帧功率会被语音包络波谷骗，与 Geigel 同类坑）；② 留余量（1.25×）；③ 真正的「不许爆」
        #    靠 **w_max 范数上限**（物理上回声路径增益 < 1），而不是每帧比较。
        sustained = self._mic_pow > 1e-9 and self._res_pow > self.bypass_ratio * self._mic_pow
        if sustained:
            self.bypass_frames += 1
            self._w *= 0.5                       # 持续更差 → 衰减权重让它重学
            e = d.copy()                         # 旁路 = 不消（原样给下游），而不是输出静音
            res_pow = mic_pow
            self._res_pow = 0.9 * self._res_pow + 0.1 * mic_pow
            diverging = False
        if near_end or diverging:
            self.frozen_frames += 1
        else:
            # 块 NLMS 归一化：分母 = **每样本输入窗功率的均值**（≈ taps × σ²）。
            # 2026-10-03 实测踩坑（三次才对）：多除一个帧长 f 就等价于每步小 f 倍（20ms 时 20 倍）
            # → 权重每帧只动万分之几，肉眼像「在跑」但 ERLE 只有 0.2dB、残差≈输入。
            # 正确形式：Δw = μ·Σ_n e[n]·x_n / mean_n(‖x_n‖²)
            p = float(np.mean(np.sum(win * win, axis=1))) + self.eps
            self._w += self.mu * (win.T @ e) / p
            # ‖w‖ 上限：参考对不上时 NLMS 会失控长大（见上）—— 夹住范数与逐点值
            nw = float(np.linalg.norm(self._w))
            if nw > self.w_max:
                self._w *= self.w_max / nw
            self._w = np.clip(self._w, -4.0, 4.0)                     # 数值兜底

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
                "bypass": self.bypass_frames, "erle_db": round(self.erle_db(), 1), "taps": self.taps}


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


class DelayTracker:
    """在线估计「麦克风听到播放声音」比「参考时间轴」晚多少毫秒。

    **为什么必须有**：真实设备的输出缓冲 + 输入缓冲 + 声学延迟合计可达数百毫秒 ——
    2026-10-03 本机实测 **520ms**，而 PortAudio 报的输入 latency 只有 26ms、任何固定常数都盖不住。
    参考对不上，AEC 就只能空转（真机实测：滤波器从未收敛、残差反而更响），残余抑制的
    「麦克风 vs 预期回声」也无从判断 → 它就把自己的话当成用户输入。

    **为什么用包络、不用波形**：笔记本喇叭→麦克风是**非线性**路径（波形相关只有 0.045），
    但**能量包络几乎不变形**（实测包络相关 0.31，峰很清晰）→ 用 20ms 的 RMS 序列做互相关最稳。

    用法：每块 `push(mic_block, ref_block)`（两段等长、时间轴各自对齐），
    定期 `estimate_ms()` 取延迟；拿到后用它去取参考（`lead_s = 延迟`）。
    """

    def __init__(self, samplerate: int = 16000, hop_ms: int = 20, max_ms: int = 1000,
                 mic_window_ms: int = 600, smooth: float = 0.5):
        self.sr = int(samplerate)
        self.hop = max(1, int(self.sr * hop_ms / 1000))
        self.hop_ms = int(hop_ms)
        self.max_hops = max(1, int(max_ms) // self.hop_ms)
        self.mic_hops = max(2, int(mic_window_ms) // self.hop_ms)
        self.smooth = float(smooth)
        self.mic_env = []
        self.ref_env = []
        self.delay_ms = None
        self.corr = 0.0
        self.samples = 0

    def push(self, mic_block, ref_block):
        """推入一块（等长）麦克风与参考音频，内部转成 20ms 的 RMS 包络。"""
        import numpy as np

        m = np.asarray(mic_block, dtype=np.float32).reshape(-1)
        r = np.asarray(ref_block, dtype=np.float32).reshape(-1)
        n = min(m.size, r.size) // self.hop
        for i in range(n):
            a, b = i * self.hop, (i + 1) * self.hop
            self.mic_env.append(float(np.sqrt(np.mean(m[a:b] ** 2))))
            self.ref_env.append(float(np.sqrt(np.mean(r[a:b] ** 2))))
            self.samples += 1
        keep = self.max_hops + self.mic_hops + 2
        if len(self.mic_env) > keep:
            del self.mic_env[:-keep]
            del self.ref_env[:-keep]

    def estimate_ms(self):
        """包络互相关求延迟（毫秒）；样本不足返回上一次的值（可能为 None）。"""
        import numpy as np

        if len(self.mic_env) < self.max_hops + self.mic_hops:
            return self.delay_ms
        mic = np.asarray(self.mic_env[-self.mic_hops:], dtype=np.float64)
        ref = np.asarray(self.ref_env[-(self.max_hops + self.mic_hops):], dtype=np.float64)
        if float(np.ptp(mic)) < 1e-6:                    # 麦克风这段没动静（没在播/太安静）→ 别乱估
            return self.delay_ms
        mc = mic - mic.mean()
        denom_m = float(np.linalg.norm(mc)) + 1e-12
        best = (-2.0, 0)
        for lag in range(0, self.max_hops + 1):
            seg = ref[len(ref) - self.mic_hops - lag: len(ref) - lag]
            if seg.size != mic.size:
                continue
            sc = seg - seg.mean()
            d = float(np.dot(mc, sc) / (denom_m * (float(np.linalg.norm(sc)) + 1e-12)))
            if d > best[0]:
                best = (d, lag)
        self.corr = best[0]
        lag_ms = best[1] * self.hop_ms
        # 相关太弱（没在播/回声太小/包络太平）→ 不敢用，保留旧值。阈值 0.25：真机非线性路径下
        # 包络相关实测 0.3 量级，而噪声段的峰值也能到 0.2 左右 —— 门槛低了会漂（实测 460→786ms）。
        if best[0] < 0.25:
            return self.delay_ms
        if self.delay_ms is not None and abs(lag_ms - self.delay_ms) > 300:
            return self.delay_ms                         # 突变 >300ms 视为一次坏估计，不采纳
        self.delay_ms = lag_ms if self.delay_ms is None else int(round(
            (1 - self.smooth) * self.delay_ms + self.smooth * lag_ms))
        return self.delay_ms

    def stats(self) -> dict:
        return {"delay_ms": self.delay_ms, "corr": round(self.corr, 3), "hops": self.samples}


class ResidualSuppressor:
    """线性 AEC 之后的**残余回声抑制**（真实产品里 AEC 后面那一步：NLP/RES）。

    **为什么必须有**：笔记本的「喇叭→麦克风」是**非线性**路径（驱动增强/削波/机壳震动），线性
    自适应滤波器在原理上建模不了 —— 2026-10-03 真机实测：麦克风与播放音频的最佳相关只有 **0.045**
    （换了 4~5 种对齐都是这个量级），于是 AEC 消不掉、它照样听得见自己（用户原话：
    「他把自己说的也当作我的输入了」）。此时唯一有效的是**按参考能量压制**。

    做法（自适应、不拍固定阈值）：播放期间用「麦克风能量 / 参考能量」的**中位数**估计耦合系数
    （中位数对个别异常帧稳健 —— 用最小值会被「回声还没到/信号很弱的帧」拉低，真实回声就被误判成
    人声而全部放行，2026-10-03 真机实测踩到），然后：

    · 麦克风能量 > `open_ratio` × 预期回声 → **有人在说话** → 原样放行（**保住插话**）
    · 否则 → 压到 `duck`（回声不再触发 VAD，它就不会听回自己）

    每轮播放重新校准：参考静音超过 `gap_ms` 视为新一轮（用户说话时段不会污染下一轮的基准）。
    """

    def __init__(self, ref_threshold: float = 1e-4, open_ratio: float = 3.0,
                 hold_ms: int = 1500, duck: float = 0.15, gap_ms: int = 300,
                 samplerate: int = 16000, min_samples: int = 4):
        self.ref_threshold = float(ref_threshold)   # 参考能量低于此 = 没在播放 → 直通
        self.open_ratio = float(open_ratio)         # 麦克风比预期回声响这么多 → 判为有人说话
        self.hold_ms = int(hold_ms)                 # 采样窗口（每轮播放开头）
        self.duck = float(duck)                     # 压制系数
        self.gap_ms = int(gap_ms)                   # 参考静音超此 → 视为新一轮
        self.sr = int(samplerate)
        self.min_samples = max(1, int(min_samples))
        self.coupling = None
        self._cal_ms = 0.0
        self.suppressed = 0
        self.passed = 0
        self.rounds = 0                             # 校准过几轮（可观测）
        self._cal = []
        self._silent_ms = 0

    def process(self, mic, ref):
        import numpy as np

        x = np.asarray(mic, dtype=np.float32)
        r = np.asarray(ref, dtype=np.float32)
        m_pow = float(np.mean(x * x)) if x.size else 0.0
        r_pow = float(np.mean(r * r)) if r.size else 0.0
        # 块长按**实际样本数**算：真实链路每块 200ms，早期版本按「每次调用 = 20ms」记账 →
        # 校准窗口被拉长 10 倍、中位数被播放中后期的安静帧拉垮（2026-10-03 真机实测踩到）
        dur_ms = (float(np.size(x)) / self.sr * 1000.0) if self.sr else 20.0
        if r_pow < self.ref_threshold:              # 没在播放：不碰麦克风（用户说话/环境声不受影响）
            self._silent_ms += dur_ms
            if self._silent_ms >= self.gap_ms:      # 静音够久 → 下一轮播放要重新校准
                self._cal, self.coupling, self._cal_ms = [], None, 0.0
            return x
        self._silent_ms = 0
        ratio = m_pow / max(r_pow, 1e-12)
        if not self._cal or self._cal_ms < self.hold_ms or len(self._cal) < self.min_samples:
            # 只让「看起来还是回声」的帧进校准：明显超出当前基准的帧可能是用户说话，
            # 让它进样本会把基准抬高 → 真实回声被误判成人声而全部放行（用例守这条）。
            if self.coupling is None or ratio <= 2.0 * self.coupling:
                self._cal.append(ratio)
                self._cal_ms += dur_ms
            self.coupling = float(np.median(self._cal))     # 边采边用（首帧就能压）
            if self._cal_ms >= self.hold_ms and len(self._cal) >= self.min_samples:
                self.rounds += 1
        if m_pow > self.open_ratio * self.coupling * r_pow:
            self.passed += 1                        # 明显比预期回声响 → 有人在说话 → 放行（保住插话）
            return x
        self.suppressed += 1
        return (x * self.duck).astype(np.float32)

    def stats(self) -> dict:
        return {"suppressed": self.suppressed, "passed": self.passed, "rounds": self.rounds,
                "coupling": round(self.coupling, 5) if self.coupling is not None else None}


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
