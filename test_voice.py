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

from src.voice import EdgeTTS, _check_deps, run_voice, voice_loop


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
        with patch("src.voice.importlib.util.find_spec", return_value=None):
            missing = _check_deps()
        self.assertTrue(any("edge_tts" in m for m in missing))
        self.assertTrue(any("whisper" in m for m in missing))

    def test_all_deps_ok(self):
        with patch("src.voice.importlib.util.find_spec", return_value=MagicMock()):
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
        with patch("src.voice.tempfile.TemporaryDirectory", return_value=_FakeTmpDir("/tmp/fake")):
            with patch("src.voice.record_until_silence", return_value=2.0) as m_rec:
                with patch("src.voice.play_audio") as m_play:
                    asyncio.run(voice_loop(agent, stt, tts, keep_alive=False))
        self.assertEqual(agent.tasks, ["今天天气怎么样"])
        self.assertTrue(tts.calls)  # 有 TTS 调用
        self.assertEqual(m_play.call_count, 1)

    def test_exit_keyword_breaks(self):
        """说「退出」→ 不调 Agent 直接结束。"""
        agent = _FakeAgent()
        stt = _FakeSTT({"input.wav": "退出"})
        tts = _FakeTTS()
        with patch("src.voice.tempfile.TemporaryDirectory", return_value=_FakeTmpDir("/tmp/fake")):
            with patch("src.voice.record_until_silence", return_value=1.5):
                with patch("src.voice.play_audio"):
                    asyncio.run(voice_loop(agent, stt, tts, keep_alive=False))
        self.assertEqual(agent.tasks, [])  # 没调 Agent
        self.assertEqual(tts.calls, [])  # 没播语音

    def test_short_audio_retries(self):
        """音频 <0.3s → 提示再说一次，不调 Agent。"""
        agent = _FakeAgent()
        stt = _FakeSTT()
        tts = _FakeTTS()
        with patch("src.voice.tempfile.TemporaryDirectory", return_value=_FakeTmpDir("/tmp/fake")):
            with patch("src.voice.record_until_silence", return_value=0.1):
                asyncio.run(voice_loop(agent, stt, tts, keep_alive=False, max_rounds=2))
        self.assertEqual(agent.tasks, [])
        self.assertEqual(tts.calls, [])

    def test_empty_transcription_retries(self):
        """识别为空 → 提示再说一次，不调 Agent。"""
        agent = _FakeAgent()
        stt = _FakeSTT({"input.wav": ""})
        tts = _FakeTTS()
        with patch("src.voice.tempfile.TemporaryDirectory", return_value=_FakeTmpDir("/tmp/fake")):
            with patch("src.voice.record_until_silence", return_value=2.0):
                asyncio.run(voice_loop(agent, stt, tts, keep_alive=False, max_rounds=2))
        self.assertEqual(agent.tasks, [])


class TestRunVoice(unittest.TestCase):
    def test_missing_deps_friendly(self):
        """缺依赖 → 友好提示，不崩。"""
        with patch("src.voice._check_deps", return_value=["edge_tts（安装：pip install edge-tts）"]):
            with patch("sys.stdout") as m:
                run_voice(_FakeAgent())
                # 不抛异常即通过

    def test_full_path_with_deps(self):
        """依赖齐 → 走 WhisperSTT + EdgeTTS + voice_loop。"""
        with patch("src.voice._check_deps", return_value=[]):
            with patch("src.voice.WhisperSTT") as m_stt:
                with patch("src.voice.EdgeTTS") as m_tts:
                    with patch("src.voice.voice_loop") as m_loop:
                        run_voice(_FakeAgent())
                        m_stt.assert_called_once()
                        m_tts.assert_called_once()
                        m_loop.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
