"""回声消除（AEC）测试 —— 离线、确定性、不需要声卡/麦克风。

覆盖三层：
1. **DSP 核心**（`NlmsAec`）：合成回声路径下的 ERLE、双讲冻结、静音参考不吃人声、分块一致性
2. **参考信号缓冲**（`ReferenceTap`）：时间轴映射（推入 → 取段）确定性正确
3. **链路级**（`MicListener` + `streaming_voice_loop`）：AEC 把「自己播回去的回声」压到不触发，
   而**真人插话照样触发**（这正是 AEC 存在的意义：不用闭麦、不用按键也能边说边插话）

2026-10-03：实现时踩到 3 个真坑（窗口切片差一格、发散保护没余量、归一化多除帧长），
都能让 AEC「看起来在跑但完全不消」——所以下面每条都有明确阈值，而不是「跑过就算」。
"""
import importlib.util
import os
import sys
import time
import unittest

try:                          # 无 numpy 环境：模块仍可被收集，用例走 skip
    import numpy as np
except ModuleNotFoundError:   # pragma: no cover
    np = None

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
needs_numpy = unittest.skipUnless(HAS_NUMPY, "AEC 测试需要 numpy（pip install numpy）")

sys.path.insert(0, ".")

from forge.aec import (NlmsAec, NullAec, ReferenceTap, ResidualSuppressor,  # noqa: E402
                       available_engines,
                       make_aec)
from forge.voice import (EnergyVAD, MicListener, NullSink,  # noqa: E402
                         SPEECH_END, ScriptedSource, streaming_voice_loop)

SR = 16000
rng = np.random.default_rng(20261003) if HAS_NUMPY else None


def echo_path(ref, delay_ms=20, gain=0.6, taps=(1.0, 0.4, 0.2)):
    """模拟「喇叭 → 空气 → 麦克风」：延迟 + 衰减 + 轻微多径（线性回声路径）。"""
    d = int(SR * delay_ms / 1000)
    out = np.zeros_like(ref)
    for i, g in enumerate(taps):
        out[d + i:] += g * gain * ref[:len(ref) - d - i]
    return out


def speech_like(n, amp=0.3):
    """带包络的噪声（当成「语音」用；AEC 只看能量/相关性，不需要真语音）。"""
    t = np.arange(n) / SR
    return (amp * rng.standard_normal(n) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.7 * t))).astype(np.float32)


@needs_numpy
class TestNlmsAec(unittest.TestCase):
    """DSP 核心：给定对齐的 参考/麦克风，看它消不消得掉、会不会误伤人声。"""

    def _run_stream(self, mic, ref, aec):
        """逐帧喂（模拟流式），返回残差。"""
        out = np.empty_like(mic)
        f = aec.frame
        for i in range(0, len(mic) - f + 1, f):
            out[i:i + f] = aec.process(mic[i:i + f], ref[i:i + f])
        return out

    def test_cancels_synthetic_echo(self):
        """稳态 ERLE 应 ≥ 15dB（实测 ~30dB）。低于 15 说明自适应没真正收敛。"""
        ref = speech_like(SR * 3)
        mic = echo_path(ref) + 0.002 * rng.standard_normal(SR * 3).astype(np.float32)
        aec = NlmsAec(samplerate=SR, frame_ms=20, filter_ms=200)
        res = self._run_stream(mic, ref, aec)
        tail = slice(-SR, None)                                  # 最后 1 秒
        erle = 10 * np.log10(np.mean(mic[tail] ** 2) / max(np.mean(res[tail] ** 2), 1e-12))
        self.assertGreater(erle, 15.0, f"ERLE 仅 {erle:.1f}dB —— 自适应没收敛")
        self.assertAlmostEqual(aec.stats()["erle_db"], erle, delta=8.0)

    def test_double_talk_freezes_and_preserves_near_end(self):
        """近端说话时冻结自适应：不把人声消掉，也不发散。"""
        ref = speech_like(SR * 3)
        mic = echo_path(ref)
        aec = NlmsAec(samplerate=SR, frame_ms=20, filter_ms=200)
        warm = int(SR * 1.5 / aec.frame) * aec.frame
        self._run_stream(mic[:warm], ref[:warm], aec)            # 先收敛
        frozen_before = aec.frozen_frames

        near = speech_like(aec.frame, amp=0.25)
        out, ins = [], []
        for i in range(warm, warm + int(SR * 0.5), aec.frame):
            d = mic[i:i + aec.frame] + near
            out.append(aec.process(d, ref[i:i + aec.frame]))
            ins.append(near)
        got, want = np.concatenate(out), np.concatenate(ins)
        self.assertGreater(aec.frozen_frames - frozen_before, 0, "双讲时应冻结自适应")
        corr = float(np.corrcoef(want, got)[0, 1])
        self.assertGreater(corr, 0.5, f"近端语音被消掉了（相关 {corr:.2f}）")

    def test_silent_reference_keeps_near_end(self):
        """没在播放（参考全零）时，人声必须原样保留 —— 否则「静音时听不见你说话」。"""
        speech = speech_like(SR // 2)
        aec = NlmsAec(samplerate=SR, frame_ms=20)
        res = self._run_stream(speech, np.zeros_like(speech), aec)
        ratio = float(np.mean(res ** 2) / np.mean(speech ** 2))
        self.assertAlmostEqual(ratio, 1.0, delta=0.05, msg=f"能量保留 {ratio:.3f}")

    def test_block_processing_matches_within_tolerance(self):
        """分块（流式）与整段处理结果应基本一致 —— 否则真实链路的延迟线是错的。"""
        ref = speech_like(SR)
        mic = echo_path(ref)
        a = self._run_stream(mic, ref, NlmsAec(samplerate=SR, frame_ms=20))
        b = NlmsAec(samplerate=SR, frame_ms=20).process(mic, ref)
        n = min(len(a), len(b))
        self.assertLess(float(np.mean((a[:n] - b[:n]) ** 2)), 1e-6)

    def test_factory_and_errors(self):
        self.assertIsInstance(make_aec("none"), NullAec)
        self.assertIsInstance(make_aec("nlms"), NlmsAec)
        self.assertIn("nlms", available_engines())
        with self.assertRaises(ValueError):
            make_aec("不存在的引擎")
        try:
            import pyaec                                          # noqa: F401
            installed = True
        except Exception:
            installed = False
        if not installed:                                          # 本机 3.13 无 wheel
            with self.assertRaises(Exception):
                make_aec("pyaec")

    def test_null_aec_is_passthrough(self):
        x = speech_like(1000)
        self.assertTrue(np.array_equal(NullAec().process(x, np.zeros(1000)), x))


@needs_numpy
class TestReferenceTap(unittest.TestCase):
    """参考信号缓冲：把「播出时刻 → 样本」的映射做对，AEC 才拿得到正确参考。"""

    def test_segment_returns_played_samples(self):
        tap = ReferenceTap(samplerate=1000, capacity_s=5.0)
        a = np.arange(100, dtype=np.float32)
        b = np.arange(100, 200, dtype=np.float32)
        tap.push(a, t=10.0)                       # 10.00s ~ 10.10s 播 a
        tap.push(b, t=10.10)                      # 10.10s ~ 10.20s 播 b
        # 取「10.20s 结束、往前 100 个样本」→ 正好是 b 的全部
        got = tap.segment(10.20, 100, lead_s=0.0)
        self.assertTrue(np.allclose(got, b))
        # 取「10.15s 结束、往前 100 个样本」→ a 的后半 + b 的前半（跨块拼接）
        got2 = tap.segment(10.15, 100, lead_s=0.0)
        self.assertTrue(np.allclose(got2[:50], a[50:]))
        self.assertTrue(np.allclose(got2[50:], b[:50]))
        # lead 生效：10.20s 时取「20ms 前结束」的 100 样本 → b 少 20 个
        got3 = tap.segment(10.20, 100, lead_s=0.02)
        self.assertTrue(np.allclose(got3, np.concatenate([a[80:], b[:80]])))

    def test_missing_time_is_silence(self):
        tap = ReferenceTap(samplerate=1000)
        self.assertTrue(np.allclose(tap.segment(5.0, 50, 0.0), np.zeros(50)))
        tap.push(np.ones(10, dtype=np.float32), t=1.0)
        got = tap.segment(1.0, 10, lead_s=0.0)                    # 完全早于任何播放
        self.assertAlmostEqual(float(np.abs(got).sum()), 0.0, places=6)


class _VClock:
    """虚拟时钟：与样本计数**严格同步**（t0 + 已读样本数/采样率）。

    用它的理由：`MicListener` 现在按「块的采集时刻」取参考，若时间戳来自真实墙钟，
    合成用例就会受机器负载影响而 flaky（2026-10-03 实测：三次跑挂一次）。替身共用同一虚拟
    时间轴后，源/汇/监听三方的时间戳完全一致 → 确定性可复现，同时保留真实节流节奏。
    """

    def __init__(self, t0=1_000_000.0):
        self.t = float(t0)
        self.t0 = float(t0)

    def __call__(self):
        return self.t
class _EchoSource:
    """假麦克风：录到的是「自己刚播出去的声音的回声」（+ 可选的人声）—— 复现外放自激。

    这是纯合成、可复现的：`tap` 里有什么就回什么（延迟 + 衰减），不需要真的喇叭和麦克风。
    """

    kind = "echo-sim"
    samplerate = SR
    block_ms = 100

    def __init__(self, tap, echo_gain=0.6, echo_delay_ms=20, near_end_at=None, near_end_len=1.0,
                 clock=None):
        self.clock = clock
        self.tap = tap
        self.echo_gain = echo_gain
        self.echo_delay_s = echo_delay_ms / 1000.0
        self.near_end_at = near_end_at            # (起始秒, 持续秒) 或 None
        self.near_end_len = near_end_len
        # 人声幅度必须**明显高于回声**：真机场景人声比回声响 20dB 以上；
        # 若测试里只高 1.4 倍，任何合理阈值都无法既压回声又保住插话（2026-10-03 踩到）。
        self.near = speech_like(int(SR * near_end_len), amp=0.9)
        self.t0 = None
        self.n = 0
        self.block_frames = int(SR * self.block_ms / 1000)

    def open(self):
        self.t0 = time.monotonic()
        return self

    def read(self, frames):
        # **实时节流**：真麦克风是阻塞读（每块耗时 = 块长）。不加节流的话，几毫秒就把参考读完了，
        # 之后麦克风只剩静音 —— 控制组就白测了（2026-10-03 踩到）。
        if self.t0 is not None:
            due = self.t0 + (self.n + frames) / SR
            gap = due - time.monotonic()
            if gap > 0:
                time.sleep(gap)
        now = time.monotonic()
        if self.clock is not None:                     # 虚拟时间轴：与已读样本数严格同步
            self.clock.t = self.clock.t0 + (self.n + frames) / SR
            now = self.clock()
        if self.t0 is None:                            # t0 必须与 now **同一时间轴**（都虚拟或都真实）
            self.t0 = now
        # 与**真实设备**一致：按「这一块的采集时刻」去参考里取当时正播出去的音频（延迟 + 衰减）。
        # 2026-10-03 改：原来用 stream()（样本计数游标）—— 那只在「假播放也按样本计数推进」时自洽，
        # 真实链路是按墙钟的；test 替身必须按墙钟建模，否则测不出参考时序类 bug（真机就是这么挂的）。
        echo = self.echo_gain * self.tap.segment(now, frames, lead_s=self.echo_delay_s)
        out = echo.copy()
        if self.near_end_at is not None:
            start, dur = self.near_end_at
            el = self.n / SR          # 用**样本计数**做时间基准：与虚拟时间轴天然一致，
            #                           不受「真实/虚拟时钟混用」影响（2026-10-03 替身踩到）
            if start <= el < start + dur:
                a = int((el - start) * SR)
                seg = self.near[a:a + frames]
                out[:seg.size] += seg
        self.n += frames
        return out.reshape(-1, 1)

    def close(self):
        pass


class _RecordingSink(NullSink):
    """模拟「播放」：把播出去的样本推给参考 tap（等价于 SoundDeviceSink 的参考通路）。"""

    def __init__(self, tap, seconds_per_play=6.0, clock=None):
        super().__init__(seconds_per_play=seconds_per_play)
        self.tap = tap
        self.clock = clock
        self._t = None

    def play(self, path):
        handle = super().play(path)
        # 用一段噪声代表「播出去的 TTS」，并**按真实播放的时间轴**分块入队（每块带它该被听到的时刻）：
        # `push(samples, t)` 支持未来时刻，因此不需要后台线程就能模拟「正在播」的时间轴。
        # 推的音频**远长于**测试时长（6s）：全量跑时负载会让假麦克风的阻塞读变慢，
        # 若只推 2s，慢读索取的窗口就会落在音频之外 → 参考变静音 → AEC 失效（用例假挂）
        pcm = speech_like(int(SR * max(self.seconds_per_play, 6.0)), amp=0.35)
        blk = max(1, int(SR * 0.1))
        t0 = self.clock.t0 if self.clock is not None else time.monotonic()
        for i in range(0, pcm.size, blk):
            self.tap.push(pcm[i:i + blk], t=t0 + i / SR)
        return handle


@needs_numpy
class TestAecInLoop(unittest.TestCase):
    """链路级断言：AEC 开着 → 回声不触发；真人插话 → 照样触发。"""

    def _drive(self, aec_on, near_end_at):
        tap = ReferenceTap(samplerate=SR, capacity_s=10.0)
        clock = _VClock()                              # 三方共用同一虚拟时间轴（确定性）
        sink = _RecordingSink(tap, seconds_per_play=2.0, clock=clock)
        src = _EchoSource(tap, near_end_at=near_end_at, clock=clock)
        aec = NlmsAec(samplerate=SR, frame_ms=20, filter_ms=120) if aec_on else None
        # 用**真实 `--aec` 的那套组合**：AEC + 残余抑制器（非线性路径上真正干活的是后者）。
        # 2026-10-03 真机结论：本机喇叭→麦克风是非线性路径，线性 AEC 消不掉，靠抑制器压住。
        sup = ResidualSuppressor(samplerate=SR) if aec_on else None
        listener = MicListener(src, EnergyVAD(threshold=0.05, silence_ms=300, min_speech_ms=200),
                               barge_ms=300, aec=aec, ref_tap=tap if aec_on else None,
                               aec_lead_ms=20, suppressor=sup, clock=clock)
        listener.start()
        # 先让它「播」一段（tap 里因此有内容 → 假麦克风会录到回声）
        played = sink.play("x.mp3")
        # 跑 2.6s：让「在线延迟估计 + 抑制器校准」都收敛，断言才稳定（1.2s 时两者都还在热身期）
        time.sleep(2.6)
        barge = listener.barge_in
        played.stop()
        listener.stop()
        return barge, sink

    def test_echo_alone_does_not_trigger_with_aec(self):
        """只有回声（没人在说话）→ AEC 开着不该判成「有人说话」。"""
        barge, _ = self._drive(aec_on=True, near_end_at=None)
        self.assertFalse(barge, "回声被 AEC 消掉后不该触发抢话")

    def test_echo_alone_does_trigger_without_aec(self):
        """对照组：同样只有回声，关掉 AEC 就会被当成插话（复现自激）。"""
        barge, _ = self._drive(aec_on=False, near_end_at=None)
        self.assertTrue(barge, "关掉 AEC 时回声应当触发（说明本用例确实在考 AEC）")

    def test_real_speech_still_barges_with_aec(self):
        """AEC 开着，但真人说话 → 必须照样被听到（否则 AEC 把用户也消了）。"""
        barge, _ = self._drive(aec_on=True, near_end_at=(1.5, 1.0))
        self.assertTrue(barge, "真人插话必须仍能触发（AEC 只该消回声）")


@needs_numpy
class TestSoundDeviceSink(unittest.TestCase):
    """`--aec` 的真实播放通路（进程内 + 参考 tap）。不打开设备、不出声，CI 安全。"""

    def test_sink_builds_with_reference_tap(self):
        """回归：`SoundDeviceSink()` 构造必须能建出参考 tap。

        2026-10-03 真 bug：voice.py 里导入时别名成 `_ReferenceTap`，构造里却写 `ReferenceTap`
        → 构造即 NameError（真跑 `--aec` 直接崩），而链路测试都自己建 tap，所以没盖到。
        """
        from forge.voice import SoundDeviceSink
        sink = SoundDeviceSink(samplerate=16000)
        self.assertIsInstance(sink.tap, ReferenceTap)
        self.assertEqual(sink.tap.samplerate, 16000)

    def test_decode_pcm_if_ffmpeg_available(self):
        """ffmpeg 解码成 float32 单声道 PCM（AEC 要样本）。没装 ffmpeg 就跳过。"""
        import shutil
        import subprocess
        import tempfile
        from forge.voice import decode_audio_pcm
        if not shutil.which("ffmpeg"):
            self.skipTest("没有 ffmpeg")
        wav = os.path.join(tempfile.mkdtemp(), "t.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                        "-t", "0.2", "-y", wav], check=True)
        pcm = decode_audio_pcm(wav, 16000)
        self.assertAlmostEqual(pcm.size, 3200, delta=400)
        self.assertEqual(pcm.dtype, np.float32)


class TestAecLeadEstimate(unittest.TestCase):
    """参考提前量按设备延迟推算（别拍常数）。2026-10-03：本机实测输入延迟 90ms，硬编码 120 差 30ms。"""

    class _Src:
        def __init__(self, lat):
            if lat is not None:
                self.latency_s = lat

    def test_uses_source_latency_plus_margin(self):
        from forge.voice import estimate_aec_lead_ms
        self.assertEqual(estimate_aec_lead_ms(self._Src(0.090)), 105)   # 90 + 15
        self.assertEqual(estimate_aec_lead_ms(self._Src(0.0)), 120)     # 0 当无效 → 保守回落
        self.assertEqual(estimate_aec_lead_ms(self._Src(0.18)), 180)    # 180+15 被上限截到 180
        self.assertEqual(estimate_aec_lead_ms(self._Src(0.001)), 40)    # 下限 40

    def test_falls_back_without_latency(self):
        from forge.voice import estimate_aec_lead_ms
        self.assertEqual(estimate_aec_lead_ms(self._Src(None)), 120)    # 文件源/假源没有延迟信息
        self.assertEqual(estimate_aec_lead_ms(self._Src(None), fallback_ms=90), 90)

    def test_margin_is_configurable(self):
        from forge.voice import estimate_aec_lead_ms
        self.assertEqual(estimate_aec_lead_ms(self._Src(0.090), margin_ms=0), 90)

    def test_sounddevice_source_open_is_idempotent(self):
        """`open()` 幂等：AEC 提前量需要在开麦后读延迟，主循环也会 open → 不能重复开流。

        （不碰真设备：塞一个哨兵流对象，再调 open() 应原样返回、不覆盖。）
        """
        from forge.voice import SoundDeviceSource
        s = SoundDeviceSource()
        sentinel = object()
        s._stream = sentinel
        self.assertIs(s.open(), s)
        self.assertIs(s._stream, sentinel)
        self.assertIsInstance(s.latency_s, float)


@needs_numpy
class TestAecSafetyNet(unittest.TestCase):
    """AEC 的硬保险：**绝不能把链路消得更差**（2026-10-03 真机声学实测抓到发散）。"""

    def test_bad_reference_cannot_blow_up(self):
        """参考对不上时不能输出满量程噪声。

        现场数据：真机声学实测里对齐估计错了 → ‖w‖ 涨到 **91.7**、残差 RMS **8.7**（比输入响 100 倍）。
        修法：‖w‖ 上限 + 「残差比输入响 → 这一帧旁路，原样输出输入」。
        """
        ref = speech_like(48000, amp=0.4)
        aec = NlmsAec(samplerate=16000, frame_ms=20, filter_ms=120)      # 120ms 跨度
        # 真实失败场景：回声延迟**超出滤波器跨度**（对齐估错 400ms）→ 滤波器追不上、梯度把它带飞
        mic = (echo_path(ref, delay_ms=400, gain=0.6) + 0.0005 * rng.standard_normal(ref.size)).astype(np.float32)
        out = aec.process(mic, ref)
        self.assertLessEqual(float(np.max(np.abs(out))), float(np.max(np.abs(mic))),
                             "硬不变量：输出峰值不得超过输入峰值（实测曾到 8.7 RMS 满量程噪声）")
        self.assertLess(float(np.sqrt(np.mean(out ** 2))), 2.0 * float(np.sqrt(np.mean(mic ** 2))),
                        "旁路保险：残差不该比输入响")
        self.assertLessEqual(float(np.linalg.norm(aec._w)), 4.0 + 1e-6, "‖w‖ 必须有上限")
        self.assertGreater(aec.bypass_frames, 0, "这种场景应触发过旁路保护")

    def test_bypass_does_not_fire_on_a_good_path(self):
        """旁路是保险，**不该在正常消回声时误触发**（否则把自己的 ERLE 吃掉）。"""
        ref = speech_like(32000, amp=0.4)
        aec = NlmsAec(samplerate=16000, frame_ms=20, filter_ms=200)
        echo = echo_path(ref, delay_ms=20, gain=0.6)
        mic = (echo + 0.0005 * rng.standard_normal(ref.size)).astype(np.float32)
        out = aec.process(mic, ref)
        # 允许极少数（≤2）单帧瞬时过冲：块更新偶尔会让某一帧残差短暂超标，旁路它一帧即可，
        # 不该因此衰减权重（真正的质量下限由下面的 ERLE 断言和 test_cancels_synthetic_echo 守着）。
        self.assertLess(aec.bypass_frames, aec.frames * 0.2, "正常回声路径不该被旁路大面积打断")
        half = out.size // 2
        erle = 10 * np.log10(float(np.mean(mic[half:] ** 2)) / max(float(np.mean(out[half:] ** 2)), 1e-12))
        self.assertGreater(erle, 15.0, f"正常路径仍应消掉 ≥15dB（实测 {erle:.1f}dB）")


@needs_numpy
class TestResidualSuppressor(unittest.TestCase):
    """残余回声抑制：喇叭→麦克风**非线性**时唯一有效的一步（AEC 之后）。

    真机实测：麦克风与播放音频最佳相关仅 0.045（线性滤波器建模不了）→ AEC 消不掉 → 它听回自己。
    """

    def _ref(self, seconds=1.0, amp=0.3):
        return (amp * np.sin(2 * np.pi * 220 * np.arange(int(16000 * seconds)) / 16000)).astype(np.float32)

    def test_echo_only_is_suppressed(self):
        """只有回声（喇叭在响、没人说话）→ 压住，别让 VAD 触发。"""
        from forge.aec import ResidualSuppressor
        ref = self._ref(1.0, 0.3)
        # 非线性回声：削波 + 噪声（真实笔记本就是这样，所以线性 AEC 无效）
        echo = np.clip(0.25 * ref, -0.2, 0.2) + 0.002 * rng.standard_normal(ref.size)
        s = ResidualSuppressor()
        out = s.process(echo.astype(np.float32), ref)
        self.assertLess(float(np.sqrt(np.mean(out ** 2))), 0.5 * float(np.sqrt(np.mean(echo ** 2))),
                        "播放期间的回声应被明显压住")
        self.assertGreater(s.suppressed, 0)

    def test_loud_near_end_passes_through(self):
        """有人明显在说话（能量远超预期回声）→ 放行，保住插话。"""
        from forge.aec import ResidualSuppressor
        ref = self._ref(1.0, 0.3)
        echo = np.clip(0.25 * ref, -0.2, 0.2)
        speech = 0.35 * (rng.standard_normal(ref.size) * np.linspace(0.5, 1.0, ref.size)).astype(np.float32)
        mic = (echo + speech).astype(np.float32)
        s = ResidualSuppressor()
        s.process(echo.astype(np.float32), ref)        # 先让基准收敛到「只有回声」
        out = s.process(mic, ref)
        self.assertGreater(float(np.sqrt(np.mean(out ** 2))), 0.8 * float(np.sqrt(np.mean(mic ** 2))),
                           "人声明显更响时应放行（否则插话被压掉）")
        self.assertGreater(s.passed, 0)

    def test_no_playback_is_passthrough(self):
        """没在播放（参考为空）→ 原样直通，绝不动用户的麦克风。"""
        from forge.aec import ResidualSuppressor
        silence = np.zeros(16000, dtype=np.float32)
        speech = (0.2 * rng.standard_normal(16000)).astype(np.float32)
        s = ResidualSuppressor()
        out = s.process(speech, silence)
        self.assertTrue(np.allclose(out, speech), "未播放时必须直通")
        self.assertEqual(s.suppressed, 0)


@needs_numpy
class TestDelayTracker(unittest.TestCase):
    """在线延迟估计（AEC 的参考必须对得上；真实设备合计延迟可达数百毫秒）。"""

    def _speechy(self, seconds, seed=3, sr=16000):
        rng = np.random.default_rng(seed)
        k = np.arange(int(seconds * sr))
        env = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2 * np.pi * k / (1.3 * sr)))
        return (env * rng.standard_normal(k.size) * 0.35).astype(np.float32)

    def test_estimates_known_delay_of_nonlinear_copy(self):
        """麦克风 = 参考延迟 600ms 的**削波**副本（模拟非线性喇叭→麦克风）→ 应估出 ≈600ms。"""
        from forge.aec import DelayTracker
        sr, blk = 16000, int(0.2 * 16000)
        ref = self._speechy(4.0)
        d = int(0.6 * sr)
        echo = np.clip(0.6 * ref, -0.5, 0.5).astype(np.float32)
        mic = np.concatenate([np.zeros(d, dtype=np.float32), echo[:ref.size - d]])
        mic = (mic + 0.002 * np.random.default_rng(9).standard_normal(mic.size)).astype(np.float32)
        tr = DelayTracker(samplerate=sr)
        for i in range(0, ref.size - blk, blk):
            tr.push(mic[i:i + blk], ref[i:i + blk])
            tr.estimate_ms()
        self.assertIsNotNone(tr.delay_ms, "应估出延迟")
        self.assertLess(abs(tr.delay_ms - 600), 60, f"估到 {tr.delay_ms}ms，期望 ≈600ms")

    def test_weak_echo_gives_a_stable_estimate(self):
        """**弱回声**（只比底噪高约 5dB）下估计必须稳定且正确。

        回归：2026-10-03 真机上单帧取峰让估计在 250/460/786ms 之间乱跳 → 参考对不上 →
        抑制器耦合基准失效 → 回声仍被当成用户输入。修法：长窗 + 跨帧累积相关谱 + 强相关才采纳。
        """
        from forge.aec import DelayTracker
        sr, blk, true_ms = 16000, int(0.2 * 16000), 520
        ref = self._speechy(9.0, seed=11)
        d = int(true_ms / 1000 * sr)
        echo = np.clip(0.6 * ref, -0.5, 0.5).astype(np.float32)
        mic = np.concatenate([np.zeros(d, dtype=np.float32), echo[:ref.size - d]])
        # 回声压到只比底噪高 ~5dB（底噪 0.006 RMS、回声 ~0.011 RMS）：真机就是这个量级
        mic = (0.006 * np.random.default_rng(4).standard_normal(mic.size)).astype(np.float32) + mic * 0.055
        tr = DelayTracker(samplerate=sr)
        ests = []
        for i in range(0, ref.size - blk, blk):
            tr.push(mic[i:i + blk], ref[i:i + blk])
            e = tr.estimate_ms()
            if e is not None:
                ests.append(e)
        self.assertTrue(ests, "弱回声下也应给出估计")
        self.assertLess(abs(ests[-1] - true_ms), 80, f"估到 {ests[-1]}ms，期望 ≈{true_ms}ms")
        tail = ests[-8:]
        self.assertLessEqual(max(tail) - min(tail), 60,
                             f"后段估计抖动 {max(tail) - min(tail)}ms（应稳定，这是本轮要修的抖动）")

    def test_silent_mic_yields_no_estimate(self):
        """麦克风全程安静（没在播/没回声）→ 不许乱给延迟（否则会把参考推错地方）。"""
        from forge.aec import DelayTracker
        sr, blk = 16000, int(0.2 * 16000)
        ref = self._speechy(3.0, seed=5)
        tr = DelayTracker(samplerate=sr)
        silent = np.zeros(blk, dtype=np.float32)
        for i in range(0, ref.size - blk, blk):
            tr.push(silent, ref[i:i + blk])
        self.assertIsNone(tr.estimate_ms())


if __name__ == "__main__":
    unittest.main(verbosity=2)
