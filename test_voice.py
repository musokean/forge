"""语音模块测试（#11 Phase 1）：TTS 合成 / 缺依赖提示 / 主循环流程 / Agent 集成。

运行：python test_voice.py  或  pytest test_voice.py
依赖：mock 覆盖录音/STT/播放（沙箱无麦克风）；Edge-TTS 需联网（可跳过）。
"""
import asyncio
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, ".")

from forge.voice import (PARTIAL, SPEECH_END, SPEECH_START, EdgeTTS, EnergyVAD, FileSource,
                       MicListener, NullSink, ScriptedSource, SentenceSplitter,
                       StreamingSpeaker, _check_deps, is_exit, run_voice,
                       streaming_voice_loop, voice_loop, write_wav)


class _FakeSTT:
    """假 STT：从预设 dict 按文件名返回文字。"""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def transcribe(self, audio_path):
        return self.mapping.get(os.path.basename(audio_path), "你好")


class _FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, out_path):
        self.calls.append((text, out_path))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("fake-mp3")
        return out_path


class _FakeAgent:
    """假 Agent：记录收到的任务，返回固定回答。"""

    def __init__(self, reply="语音回答成功"):
        self.reply = reply
        self.tasks = []

    async def run(self, task):
        self.tasks.append(task)
        return self.reply


class TestDepsCheck(unittest.TestCase):
    def test_missing_deps_reported(self):
        with patch("forge.voice.importlib.util.find_spec", return_value=None):
            missing = _check_deps()
        self.assertTrue(any("edge_tts" in m for m in missing))
        self.assertTrue(any("whisper" in m for m in missing))

    def test_all_deps_ok(self):
        with patch("forge.voice.importlib.util.find_spec", return_value=MagicMock()):
            self.assertEqual(_check_deps(), [])


class TestTTSSynthesis(unittest.TestCase):
    """Edge-TTS 真实合成（需联网；断网/失败时跳过不阻塞）。"""

    def test_edge_tts_synthesizes_mp3(self):
        try:
            tts = EdgeTTS()
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "t.mp3")
                tts.synthesize("测试语音合成", out)
                self.assertTrue(os.path.exists(out))
                self.assertGreater(os.path.getsize(out), 1000)  # mp3 非空
        except Exception as e:
            self.skipTest(f"Edge-TTS 不可用（网络/依赖）：{e}")


class _FakeTmpDir:
    """可 with 的临时目录假对象。"""

    def __init__(self, path):
        self._p = path

    def __enter__(self):
        return self._p

    def __exit__(self, *a):
        return False


class TestVoiceLoop(unittest.TestCase):
    def test_normal_flow(self):
        """说一句 → 识别 → Agent 回答 → TTS 播放。"""
        agent = _FakeAgent()
        stt = _FakeSTT({"input.wav": "今天天气怎么样"})
        tts = _FakeTTS()
        with patch("forge.voice.tempfile.TemporaryDirectory", return_value=_FakeTmpDir("/tmp/fake")):
            with patch("forge.voice.record_until_silence", return_value=2.0) as m_rec:
                with patch("forge.voice.play_audio") as m_play:
                    asyncio.run(voice_loop(agent, stt, tts, keep_alive=False))
        self.assertEqual(agent.tasks, ["今天天气怎么样"])
        self.assertTrue(tts.calls)  # 有 TTS 调用
        self.assertEqual(m_play.call_count, 1)

    def test_exit_keyword_breaks(self):
        """说「退出」→ 不调 Agent 直接结束。"""
        agent = _FakeAgent()
        stt = _FakeSTT({"input.wav": "退出"})
        tts = _FakeTTS()
        with patch("forge.voice.tempfile.TemporaryDirectory", return_value=_FakeTmpDir("/tmp/fake")):
            with patch("forge.voice.record_until_silence", return_value=1.5):
                with patch("forge.voice.play_audio"):
                    asyncio.run(voice_loop(agent, stt, tts, keep_alive=False))
        self.assertEqual(agent.tasks, [])  # 没调 Agent
        self.assertEqual(tts.calls, [])  # 没播语音

    def test_short_audio_retries(self):
        """音频 <0.3s → 提示再说一次，不调 Agent。"""
        agent = _FakeAgent()
        stt = _FakeSTT()
        tts = _FakeTTS()
        with patch("forge.voice.tempfile.TemporaryDirectory", return_value=_FakeTmpDir("/tmp/fake")):
            with patch("forge.voice.record_until_silence", return_value=0.1):
                asyncio.run(voice_loop(agent, stt, tts, keep_alive=False, max_rounds=2))
        self.assertEqual(agent.tasks, [])
        self.assertEqual(tts.calls, [])

    def test_empty_transcription_retries(self):
        """识别为空 → 提示再说一次，不调 Agent。"""
        agent = _FakeAgent()
        stt = _FakeSTT({"input.wav": ""})
        tts = _FakeTTS()
        with patch("forge.voice.tempfile.TemporaryDirectory", return_value=_FakeTmpDir("/tmp/fake")):
            with patch("forge.voice.record_until_silence", return_value=2.0):
                asyncio.run(voice_loop(agent, stt, tts, keep_alive=False, max_rounds=2))
        self.assertEqual(agent.tasks, [])


class TestRunVoice(unittest.TestCase):
    def test_missing_deps_friendly(self):
        """缺依赖 → 友好提示，不崩。"""
        with patch("forge.voice._check_deps", return_value=["edge_tts（安装：pip install edge-tts）"]):
            with patch("sys.stdout") as m:
                run_voice(_FakeAgent())
                # 不抛异常即通过

    def test_full_path_with_deps(self):
        """依赖齐 → 走 WhisperSTT + EdgeTTS + **Phase 2/3 流式循环**（默认）。"""
        with patch("forge.voice._check_deps", return_value=[]):
            with patch("forge.voice.WhisperSTT") as m_stt:
                with patch("forge.voice.EdgeTTS") as m_tts:
                    with patch("forge.voice.streaming_voice_loop") as m_loop:
                        run_voice(_FakeAgent())
                        m_stt.assert_called_once()
                        m_tts.assert_called_once()
                        m_loop.assert_called_once()

    def test_phase1_path_still_available(self):
        """stream=False → 退回 Phase 1 的 voice_loop（保留的老链路）。"""
        with patch("forge.voice._check_deps", return_value=[]):
            with patch("forge.voice.WhisperSTT"):
                with patch("forge.voice.EdgeTTS"):
                    with patch("forge.voice.voice_loop") as m_loop:
                        run_voice(_FakeAgent(), stream=False)
                        m_loop.assert_called_once()

    def test_file_source_needs_no_microphone(self):
        """--audio-source file:xx.wav → 不检查 sounddevice（无麦克风也能跑）。"""
        with patch("forge.voice._check_deps", return_value=["sounddevice"]):
            with patch("forge.voice.WhisperSTT"):
                with patch("forge.voice.EdgeTTS"):
                    with patch("forge.voice.streaming_voice_loop") as m_loop:
                        run_voice(_FakeAgent(), audio_source="file:/tmp/x.wav", sink="null")
                        self.assertTrue(m_loop.called)
                        src = m_loop.call_args.kwargs.get("source")
                        self.assertIsInstance(src, FileSource)


# ══════════════════════════════════════════════════════════════════════
# #11 Phase 2/3（流式 + 打断）—— 全离线：不需要真麦克风、不需要模型、不出声
#
# 能这么测，是因为语音链路被拆成了可注入的三块：
#   AudioSource（真麦克风 / 文件 / 脚本化合成音频）· AudioSink（真播放 / 只记录）
#   VAD 与切句是纯状态机 → 「分段边界 / 打断时序 / 流式顺序」都能确定性复现
# ══════════════════════════════════════════════════════════════════════
import importlib.util  # noqa: E402
import time  # noqa: E402

# #11 语音测试要合成音频块（VAD/切句/打断都吃 numpy 数组）：CI 里装了 numpy 会真跑，
# 没装的精简环境跳过而不是报错（同 fastapi/sounddevice 的处理方式）
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
needs_numpy = unittest.skipUnless(HAS_NUMPY, "语音流式/打断测试需要 numpy（pip install numpy）")


class _ScriptAgent:
    """假 Agent：模拟「边生成边回调增量」，并在生成期间周期询问 interrupt_check。"""

    def __init__(self, answer="第一句。第二句。第三句。", chunk=6, delay=0.0):
        self.answer = answer
        self.chunk = chunk
        self.delay = delay
        self.calls = []
        self.interrupted = False

    async def run(self, task, on_delta=None, interrupt_check=None):
        self.calls.append(task)
        for i in range(0, len(self.answer), self.chunk):
            if interrupt_check is not None and interrupt_check():
                self.interrupted = True
                return self.answer[:i] + "（被打断）"
            if on_delta:
                on_delta(self.answer[i:i + self.chunk])
            if self.delay:
                await asyncio.sleep(self.delay)
        return self.answer


class _LegacyAgent:
    """老接口（不接受 on_delta / interrupt_check）→ 循环必须优雅退化成整段合成播放。"""

    async def run(self, task):
        return "整段答案，没有增量。第二句。"


class _SeqSTT:
    """按顺序返回预设结果的假 STT（finals 用于整句，partials 用于草稿）。"""

    def __init__(self, finals=None, partials=None):
        self.finals = list(finals or ["你好"])
        self.partials = list(partials or ["你", "你好"])
        self.final_calls = 0
        self.partial_calls = 0

    def transcribe(self, audio):
        self.final_calls += 1
        return self.finals.pop(0) if self.finals else ""

    def transcribe_partial(self, audio):
        self.partial_calls += 1
        return self.partials.pop(0) if self.partials else ""


class _QuietTTS:
    """不联网的假 TTS：写个占位文件就行（NullSink 不看内容，只看时长）。"""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.texts = []

    def synthesize(self, text, out_path):
        self.texts.append(text)
        if self.delay:
            time.sleep(self.delay)
        with open(out_path, "wb") as f:
            f.write(b"fake-mp3")
        return out_path


def _wait_for(pred, timeout=3.0, interval=0.01):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(interval)
    return False


@needs_numpy
class TestEnergyVAD(unittest.TestCase):
    """VAD 是纯状态机：喂合成的块就能验边界，不需要真声音。"""

    def _vad(self, **kw):
        kw.setdefault("threshold", 0.05)
        kw.setdefault("silence_ms", 300)
        kw.setdefault("min_speech_ms", 200)
        kw.setdefault("partial_ms", 1000)
        return EnergyVAD(**kw)

    def _block(self, loud):
        import numpy as np
        return np.full(1600, 0.3 if loud else 0.0, dtype="float32")

    def test_silence_produces_nothing(self):
        vad = self._vad()
        events = []
        for _ in range(5):
            events += vad.feed(self._block(False), 100)
        self.assertEqual(events, [])
        self.assertFalse(vad.speaking)

    def test_speech_start_then_end(self):
        vad = self._vad()
        events = []
        for _ in range(3):
            events += vad.feed(self._block(True), 100)
        self.assertEqual(events[0].kind, SPEECH_START)
        self.assertTrue(vad.speaking)
        for _ in range(4):
            events += vad.feed(self._block(False), 100)
        self.assertEqual([e.kind for e in events][-1], SPEECH_END)
        self.assertFalse(vad.speaking)

    def test_short_blip_dropped(self):
        """100ms 的响声（咳嗽/敲键盘）短于 min_speech_ms → 不当成一整句话。"""
        vad = self._vad()
        vad.feed(self._block(True), 100)
        events = []
        for _ in range(4):
            events += vad.feed(self._block(False), 100)
        self.assertNotIn(SPEECH_END, [e.kind for e in events])

    def test_partial_cadence(self):
        vad = self._vad(partial_ms=300)
        events = []
        for _ in range(10):
            events += vad.feed(self._block(True), 100)
        self.assertGreaterEqual([e.kind for e in events].count(PARTIAL), 2)

    def test_noise_below_threshold_ignored(self):
        vad = self._vad()
        import numpy as np
        events = []
        for _ in range(5):
            events += vad.feed(np.full(1600, 0.01, dtype="float32"), 100)
        self.assertEqual(events, [])


class TestSentenceSplitter(unittest.TestCase):
    """切句：让 TTS 能「一句合成一句」——切错了会念得支离破碎。"""

    def test_basic_split(self):
        s = SentenceSplitter(min_chars=2)
        out = s.feed("你好。今天天气")
        self.assertEqual(out, ["你好。"])
        out2 = s.feed("不错。")
        self.assertEqual(out2, ["今天天气不错。"])

    def test_min_chars_merges_short_fragments(self):
        s = SentenceSplitter(min_chars=4)
        self.assertEqual(s.feed("好。"), [], "只有 2 字，不够 min_chars 先攒着")
        out = s.feed("我很好。")
        self.assertEqual(out, ["好。我很好。"], "攒够长度后连标点一起吐出")

    def test_max_chars_force_split(self):
        s = SentenceSplitter(min_chars=4, max_chars=20)
        out = s.feed("这是一段很长很长的句子，" * 4)
        self.assertTrue(out)
        self.assertTrue(all(len(x) <= 22 for x in out), out)

    def test_flush_returns_remainder(self):
        s = SentenceSplitter(min_chars=50)
        self.assertEqual(s.feed("没说完的一句"), [])
        self.assertEqual(s.flush(), ["没说完的一句"])

    def test_decimal_point_not_split(self):
        s = SentenceSplitter(min_chars=2)
        out = s.feed("总共 3.14 秒。结束了。")
        self.assertEqual(len(out), 2)
        self.assertIn("3.14", out[0])

    def test_clean_for_tts(self):
        s = SentenceSplitter(min_chars=2)
        raw = "**重点**：用 `forge` 看 [文档](https://x.com/a) 就好。"
        out = s.feed(raw)
        self.assertTrue(out)
        joined = out[0]
        for bad in ("**", "`", "](", "https://"):
            self.assertNotIn(bad, joined)


class TestStreamingSpeaker(unittest.TestCase):
    """句级流水线播放器：顺序不能乱，stop() 必须立刻停。"""

    def test_plays_in_order(self):
        tts, sink = _QuietTTS(), NullSink(audio_len=lambda p: 0.02)
        sp = StreamingSpeaker(tts, sink).start()
        for s in ("第一句。", "第二句。", "第三句。"):
            sp.say(s)
        self.assertTrue(sp.finish(timeout=5))
        self.assertEqual(tts.texts, ["第一句。", "第二句。", "第三句。"])
        self.assertEqual(len(sink.played), 3)
        sp.close()
    def test_stop_before_start_does_not_silence_next_turn(self):
        """回归：线程还没起来时被 `stop()` 过，不该让下一轮变哑。

        2026-09-29 抓到的真 bug：`stop()` 往队列塞哨兵 `None` → 下一轮 `start()` 起线程后
        立刻吃到哨兵退出 → 整段回答无声（`sink.played == 0`）。PTT 在**第一轮之前**按一下键
        就会走到这个时序，所以它不是一个只在测试里才会出现的边角。
        """
        sink = NullSink(audio_len=lambda p: 0.05)
        speaker = StreamingSpeaker(_QuietTTS(), sink)
        speaker.stop()                       # 线程尚未启动就打断（PTT 首按的时序）
        speaker.start()                      # 新一轮
        speaker.say("第一句。")
        self.assertTrue(speaker.finish(timeout=2.0), "队列应能排空（工作线程没被哨兵毒死）")
        self.assertEqual(len(sink.played), 1, "这一句必须真的播出去")
        speaker.close()


    def test_stop_interrupts_and_drops_queue(self):
        tts, sink = _QuietTTS(), NullSink(audio_len=lambda p: 0.6)   # 每句"播"0.6s
        sp = StreamingSpeaker(tts, sink).start()
        for s in ("第一句。", "第二句。", "第三句。"):
            sp.say(s)
        self.assertTrue(_wait_for(lambda: len(sink.played) >= 1, timeout=3))
        sp.stop()
        time.sleep(0.2)
        self.assertTrue(sink.stops, "stop() 必须打断当前播放")
        self.assertLessEqual(len(sink.played), 2, "未播的句子应被丢弃")
        sp.close()

    def test_finish_waits_for_drain(self):
        tts, sink = _QuietTTS(), NullSink(audio_len=lambda p: 0.05)
        sp = StreamingSpeaker(tts, sink).start()
        sp.say("一句话。")
        self.assertFalse(sp.finish(timeout=0.001))
        self.assertTrue(sp.finish(timeout=3))
        sp.close()


@needs_numpy
class TestMicListener(unittest.TestCase):
    """说话期间仍在听的麦克风线程：产出事件、判定抢话、保住已说半句。"""

    def _listener(self, script, barge_ms=300, **vad_kw):
        vad_kw.setdefault("threshold", 0.05)
        vad_kw.setdefault("silence_ms", 300)
        vad_kw.setdefault("min_speech_ms", 200)
        src = ScriptedSource(script, block_ms=100, amp=0.3, pace=0.01)
        return MicListener(src, EnergyVAD(**vad_kw), barge_ms=barge_ms)

    def test_events_and_utterance_captured(self):
        lis = self._listener([("speech", 0.8), ("silence", 0.8)]).start()
        try:
            self.assertTrue(_wait_for(lambda: any(
                e.kind == SPEECH_END for e in lis.drain_events()), timeout=4))
            audio = lis.take_utterance()
            self.assertIsNotNone(audio)
            self.assertGreater(len(audio), 0.8 * lis.source.samplerate, "应保住整句音频")
        finally:
            lis.stop()

    def test_barge_in_flag(self):
        lis = self._listener([("speech", 0.8), ("silence", 0.8)], barge_ms=300).start()
        try:
            self.assertTrue(_wait_for(lambda: lis.barge_in, timeout=4))
            self.assertTrue(lis(), "MicListener 本身要能当 interrupt_check 用")
        finally:
            lis.stop()

    def test_barge_threshold_respected(self):
        """抢话阈值高于说话时长 → 不该误判抢话。"""
        lis = self._listener([("speech", 0.3), ("silence", 0.8)], barge_ms=2000).start()
        try:
            time.sleep(0.8)
            self.assertFalse(lis.barge_in)
        finally:
            lis.stop()

    def test_utterance_audio_not_flooded_when_loop_is_slow(self):
        """源读得快 + 主循环忙（真 STT 要秒级）时，缓冲区不能被后续静音灌爆。

        2026-09-29 实测踩到：第二次草稿拿到 **63 秒**音频（几乎全是静音）→ whisper 识别为空。
        修法是 SPEECH_END 时把这一句冻结下来，后续帧进新缓冲。
        """
        src = ScriptedSource([("speech", 0.6), ("silence", 6.0)], block_ms=100, amp=0.3, pace=0.0)
        lis = MicListener(src, EnergyVAD(threshold=0.05, silence_ms=300, min_speech_ms=200),
                          barge_ms=300).start()
        try:
            self.assertTrue(_wait_for(
                lambda: any(e.kind == SPEECH_END for e in lis.drain_events()), timeout=3))
            time.sleep(0.3)          # 模拟主循环正忙于转写
            audio = lis.take_utterance()
            secs = len(audio) / lis.source.samplerate
            self.assertLess(secs, 1.6, f"句尾之后不该继续灌进这一句（实得 {secs:.1f}s）")
            self.assertGreater(secs, 0.6, "这句话本身要完整保住")
        finally:
            lis.stop()

    def test_clear_resets_state(self):
        lis = self._listener([("speech", 0.6), ("silence", 0.6)]).start()
        try:
            self.assertTrue(_wait_for(lambda: lis.barge_in, timeout=3))
            lis.clear()
            self.assertFalse(lis.barge_in)
            self.assertEqual(lis.peek_seconds(), 0.0)
        finally:
            lis.stop()


@needs_numpy
class TestStreamingVoiceLoop(unittest.TestCase):
    """Phase 2/3 主循环：流式转写 → 句级播放 → 播放/生成期间可抢话。"""

    def _run(self, script, stt, agent, rounds=1, sink=None, vad_kw=None, barge_ms=300,
             timeout_s=6.0, **kw):
        vad_kw = vad_kw or {}
        vad_kw.setdefault("threshold", 0.05)
        vad_kw.setdefault("silence_ms", 300)
        vad_kw.setdefault("min_speech_ms", 200)
        src = ScriptedSource(script, block_ms=100, amp=0.3, pace=0.1 if kw.pop("realtime", False) else 0.005)
        listener = MicListener(src, EnergyVAD(**vad_kw), barge_ms=barge_ms)
        events = []
        stats = asyncio.run(streaming_voice_loop(
            agent, stt, _QuietTTS(),
            listener=listener,
            sink=sink or NullSink(audio_len=lambda p: 0.05),
            max_rounds=rounds, timeout_s=timeout_s, barge_ms=barge_ms,
            on_event=events.append))
        return stats, events, listener

    def test_single_turn_streams_sentences_to_speaker(self):
        sink = NullSink(audio_len=lambda p: 0.05)
        stats, events, _ = self._run(
            [("speech", 0.6), ("silence", 1.0)], _SeqSTT(finals=["你好"]), _ScriptAgent(),
            sink=sink)
        self.assertEqual(stats["rounds"], 1)
        self.assertEqual(stats["transcripts"], ["你好"])
        kinds = [e.kind for e in events]
        self.assertIn("user", kinds)
        self.assertGreaterEqual(kinds.count("sentence"), 2)
        self.assertGreaterEqual(len(sink.played), 2, "应逐句播放，而不是整段一次")

    def test_exit_word_ends_loop(self):
        stats, events, _ = self._run(
            [("speech", 0.6), ("silence", 1.0)], _SeqSTT(finals=["退出"]), _ScriptAgent())
        self.assertIn("exit", [e.kind for e in events])
        self.assertEqual(stats["rounds"], 0)

    def test_too_short_audio_ignored(self):
        """100ms 的响声过了 min_speech 门槛但太短 → 不当作一句话（防误触发）。"""
        stats, events, _ = self._run(
            [("speech", 0.1), ("silence", 1.0)], _SeqSTT(finals=["喂"]), _ScriptAgent(),
            vad_kw={"min_speech_ms": 100}, timeout_s=1.5)
        self.assertEqual(stats["rounds"], 0)
        self.assertIn("too_short", [e.kind for e in events])

    def test_partial_transcripts_streamed(self):
        stt = _SeqSTT(finals=["请帮我算个题"], partials=["请", "请帮我", "请帮我算"])
        stats, events, _ = self._run(
            [("speech", 2.0), ("silence", 1.0)], stt, _ScriptAgent(),
            vad_kw={"partial_ms": 300})
        self.assertGreaterEqual([e.kind for e in events].count("partial"), 2)
        self.assertGreaterEqual(stt.partial_calls, 2)

    def test_barge_in_during_playback_stops_and_reuses_half_sentence(self):
        """**Phase 3 最常用的场景**：forge 正说着，用户插话 → 立刻停播 + 已说半句进下一轮。"""
        sink = NullSink(audio_len=lambda p: 0.5)          # 每句播 0.5s，留出被打断的窗口
        stt = _SeqSTT(finals=["第一个问题", "第二个问题"])
        agent = _ScriptAgent(answer="第一句。第二句。第三句。", chunk=4, delay=0.05)
        stats, events, _ = self._run(
            [("speech", 0.6), ("silence", 0.5),      # 第一轮：说完 → 生成 → 开始播
             ("speech", 0.5), ("silence", 0.3),      # ← 播放期间插话（≥barge_ms）
             ("speech", 1.0), ("silence", 1.0)],     # 抢话后的那半句 + 第二轮说完
            stt, agent, rounds=2, sink=sink, realtime=True, timeout_s=12.0)
        self.assertGreaterEqual(stats["barge_ins"], 1, "应记录到抢话")
        self.assertTrue(sink.stops, "抢话要立刻停播（不能等这句念完）")
        self.assertGreaterEqual(len(stats["transcripts"]), 2,
                                "抢话时已说的半句要保住并当成下一轮输入，不用用户再说一遍")

    def test_barge_in_cancels_generation(self):
        """生成过程中插话 → 走 interrupt_check **取消当前生成**（返回已生成部分）。"""
        stt = _SeqSTT(finals=["讲个长故事"])
        agent = _ScriptAgent(answer="第一句。第二句。第三句。第四句。", chunk=3, delay=0.3)
        stats, events, _ = self._run(
            [("speech", 0.6), ("silence", 0.5),      # 说完 → 进入生成（生成要 1.2s 左右）
             ("speech", 0.5), ("silence", 0.6)],     # ← 生成期间插话
            stt, agent, rounds=1, realtime=True, timeout_s=8.0)
        self.assertTrue(agent.interrupted, "生成期间的抢话要经过 interrupt_check 取消生成")
        self.assertTrue(stats["answers"], "被取消也要留下已生成的部分")
        self.assertIn("被打断", stats["answers"][0])
        self.assertGreaterEqual(stats["barge_ins"], 1)

    def test_legacy_agent_without_kwargs_falls_back(self):
        """老 Agent（不支持 on_delta/interrupt_check）→ 退化成整段合成播放，不崩。"""
        sink = NullSink(audio_len=lambda p: 0.02)
        stats, events, _ = self._run(
            [("speech", 0.6), ("silence", 1.0)], _SeqSTT(finals=["你好"]), _LegacyAgent(),
            sink=sink)
        self.assertEqual(stats["rounds"], 1)
        self.assertGreaterEqual(len(sink.played), 1)

    def test_is_exit_words(self):
        for w in ("退出", "exit", "结束", "再见", "拜拜。", "EXIT"):
            self.assertTrue(is_exit(w), w)
        for w in ("不要退出啊", "继续", "重置"):
            self.assertFalse(is_exit(w), w)


@needs_numpy
class TestFileSource(unittest.TestCase):
    """文件当麦克风：不用真麦克风也能跑完整链路（L2）。"""

    def test_reads_blocks_and_pads_silence(self):
        import numpy as np
        tmp = tempfile.mkdtemp()
        wav = os.path.join(tmp, "t.wav")
        write_wav(wav, np.full(16000, 0.3, dtype="float32"), 16000)      # 1s 有声
        src = FileSource(wav, block_ms=200).open()
        blocks = [src.read(src.block_frames) for _ in range(7)]          # 1.4s 的块
        self.assertEqual(blocks[0].shape[1], 1)
        self.assertGreater(float(np.sqrt((blocks[0] ** 2).mean())), 0.1)
        self.assertLess(float(np.sqrt((blocks[-1] ** 2).mean())), 0.01, "读完后应补静音而不是崩")
        src.close()


class ScriptedGate:
    """假的「按住说话」状态源（PTT 测试用；真键盘是 `KeyHold`）。

    spans = [(起始秒, 是否按住), ...] 按时间升序 —— 时间一到就切换，
    这样「按下 → 说 → 松开」的时序在 CI 里可复现，不需要真按键。
    """

    def __init__(self, spans=None, calls=None):
        self.spans = spans                      # 时间驱动：[(起始秒, 是否按住), ...]
        self.calls = calls                      # 轮询次数驱动：[bool, ...]（末尾保持）
        self.n = 0
        self.t0 = None

    def held(self):
        if self.calls is not None:              # 确定性模式：不受机器负载影响
            i = min(self.n, len(self.calls) - 1)
            self.n += 1
            return bool(self.calls[i])
        if self.t0 is None:
            self.t0 = time.time()
        now = time.time() - self.t0
        held = False
        for s, h in self.spans:
            if now >= s:
                held = h
        return held


@needs_numpy
class TestNoHeadphoneModes(unittest.TestCase):
    """**不用耳机也能用**：半双工（播放时闭麦）与按住说话（PTT）。

    背景：外放时喇叭里的 TTS 会被自己的麦克风听回去 → 被当成新指令 → 自激（2026-09-29 用户问）。
    方案：① 半双工 = 它说话时闭麦，防自激（代价：播放时不能插话，生成时仍可插话）；
         ② PTT = 不按不采集，按下即打断、松开即提交（完全不用耳机也能随时插话）。
    """

    def _run(self, script, stt, agent, rounds=1, sink=None, vad_kw=None, barge_ms=300,
             timeout_s=6.0, half_duplex=False, ptt=None, pace=0.1):
        vad_kw = dict(vad_kw or {})
        vad_kw.setdefault("threshold", 0.05)
        vad_kw.setdefault("silence_ms", 300)
        vad_kw.setdefault("min_speech_ms", 200)
        src = ScriptedSource(script, block_ms=100, amp=0.3, pace=pace)
        listener = MicListener(src, EnergyVAD(**vad_kw), barge_ms=barge_ms)
        events = []
        stats = asyncio.run(streaming_voice_loop(
            agent, stt, _QuietTTS(), listener=listener,
            sink=sink or NullSink(audio_len=lambda p: 0.05),
            max_rounds=rounds, timeout_s=timeout_s, barge_ms=barge_ms,
            half_duplex=half_duplex, guard_ms=0, ptt=ptt, on_event=events.append))
        return stats, events, listener

    # ── 半双工（播放期间闭麦）──
    def test_half_duplex_does_not_hear_itself(self):
        """它说话时喇叭声被麦克风收回去 —— 半双工下不该被当成插话、也不该多出一轮。"""
        sink = NullSink(audio_len=lambda p: 0.5)          # 每句播 0.5s，播放窗口够长
        stt = _SeqSTT(finals=["第一个问题", "这是回声不该被识别"])
        agent = _ScriptAgent(answer="第一句。第二句。第三句。第四句。", chunk=4, delay=0.02)
        stats, events, _ = self._run(
            [("speech", 0.6), ("silence", 0.4),
             ("speech", 1.5), ("silence", 0.8)],         # ← 落在「播放中」＝回声
            stt, agent, sink=sink, half_duplex=True, timeout_s=4.0)
        self.assertEqual(stats["transcripts"], ["第一个问题"], "播放期间的输入（回声）不该进第二轮")
        self.assertEqual(stats["barge_ins"], 0, "闭麦期间不该判抢话")
        self.assertEqual(stt.final_calls, 1, "只该转写一次（用户那一句）")

    def test_half_duplex_still_interrupts_while_generating(self):
        """生成期间（喇叭还没出声、不存在回声）插话仍然能打断 —— 半双工只闭「播放」那一段。"""
        sink = NullSink(audio_len=lambda p: 0.2)
        stt = _SeqSTT(finals=["第一个问题"])
        agent = _ScriptAgent(answer="一句。二句。三句。四句。五句。", chunk=3, delay=0.35)
        stats, events, _ = self._run(
            [("speech", 0.6), ("silence", 0.4),
             ("speech", 0.6), ("silence", 0.3)],
            stt, agent, sink=sink, half_duplex=True, timeout_s=5.0)
        self.assertGreaterEqual(stats["barge_ins"], 1, "生成期的插话应该仍然生效")

    def test_muted_listener_drops_audio(self):
        """闭麦 = 当没听见：不产生事件、也不攒音频；开麦后恢复检测。"""
        # 实时速度（pace=0.1）：否则源会被瞬间读完，开麦后已没有声音可检测
        src = ScriptedSource([("speech", 2.0), ("silence", 0.5)], block_ms=100, amp=0.3, pace=0.1)
        lis = MicListener(src, EnergyVAD(threshold=0.05, silence_ms=300, min_speech_ms=200)).start()
        lis.set_muted(True)
        time.sleep(0.4)
        self.assertEqual(lis.drain_events(), [], "闭麦期间不该有事件")
        self.assertEqual(len(lis.peek_audio() or []), 0, "闭麦期间不该攒音频")
        lis.set_muted(False)
        got = []
        self.assertTrue(_wait_for(lambda: got.extend(lis.drain_events()) or
                                  any(e.kind == SPEECH_START for e in got), timeout=2.0),
                        "开麦后应重新开始检测")
        lis.stop()

    # ── 按住说话（PTT）──
    def test_ptt_hold_produces_one_turn(self):
        """按住一次 = 一轮对话；不按的那段时间（哪怕有声音）不算输入。"""
        sink = NullSink(audio_len=lambda p: 0.05)
        stt = _SeqSTT(finals=["按键说的话"])
        agent = _ScriptAgent(answer="好的。", chunk=3, delay=0.0)
        gate = ScriptedGate([(0.4, True), (1.3, False)])
        stats, events, _ = self._run(
            [("silence", 0.4), ("speech", 0.8), ("silence", 1.2)],
            stt, agent, sink=sink, ptt=gate, timeout_s=4.0)
        kinds = [e.kind for e in events]
        self.assertIn("ptt_down", kinds)
        self.assertIn("ptt_up", kinds)
        self.assertEqual(stats["rounds"], 1, "按住一次 = 一轮")
        self.assertEqual(stats["transcripts"], ["按键说的话"])
        self.assertEqual(stt.final_calls, 1)

    def test_ptt_press_interrupts_playback(self):
        """按下 = 立刻打断它正在说的（这就是免耳机时的「插话」）。"""
        sink = NullSink(audio_len=lambda p: 1.2)          # 每句播 1.2s，播放窗口留足
        stt = _SeqSTT(finals=["第一个问题", "插话内容"])
        agent = _ScriptAgent(answer="第一句话在这里。第二句话在这里。", chunk=6, delay=0.0)
        # PTT 模式下**第一轮也要按住**（不按不采集）：0.3-1.1s 说第一句 → 它开始回答并播；
        # 1.7s 再按住 → 这时正在播 → 应该立刻停（rounds=2 才有第二次按住的时机）
        gate = ScriptedGate([(0.3, True), (1.1, False), (1.7, True), (2.6, False)])
        stats, events, _ = self._run(
            [("silence", 0.2), ("speech", 0.8), ("silence", 0.6), ("speech", 0.6), ("silence", 1.0)],
            stt, agent, rounds=2, sink=sink, ptt=gate, timeout_s=6.0)
        kinds = [e.kind for e in events]
        self.assertEqual(kinds.count("ptt_down"), 2, "两次按住")
        self.assertTrue(sink.stops, "按下时应立刻停播")

    def test_ptt_too_short_ignored(self):
        """误碰一下（0.15s）不该触发一轮。"""
        # 按**轮询次数**驱动（而非墙钟）：否则负载重时整段按住可能落在两次轮询之间 → flaky
        gate = ScriptedGate(calls=[False, False, True, True, False])
        stats, events, _ = self._run(
            [("silence", 1.6)], _SeqSTT(finals=["不该出现"]), _ScriptAgent(),
            ptt=gate, timeout_s=2.5)
        self.assertEqual(stats["rounds"], 0)
        self.assertIn("too_short", [e.kind for e in events])


if __name__ == "__main__":
    unittest.main(verbosity=2)
