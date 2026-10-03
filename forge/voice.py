"""#11 语音交互 · Phase 1（整句级）+ **Phase 2/3（流式 + 打断）**。

链路（级联三段，A18）:  麦克风 → VAD 分段 → STT → Agent → TTS → 播放

**Phase 1（旧）**：说一句 → 静音自动结束 → 整段转写 → 答完 → 整段合成 → 播放。
`record_until_silence` / `voice_loop` / `run_voice` 保留可用（不再往下删）。

**Phase 2 流式**：
  · 转写流式：说话期间按 `partial_ms` 周期性把「已采集到的音频」送去转写，先出草稿（`on_event("partial")`）
  · 应答流式：Agent 边生成边把增量喂给 `SentenceSplitter` 切句，**合成一句就播一句**（不等整段答案）
    → 首字出声延迟从「整段生成 + 整段合成」降到「首句生成 + 首句合成」

**Phase 3 打断（barge-in）**：forge 说话（或正在生成）期间麦克风**仍在听**；
  · 检测到用户持续说话 ≥ `barge_ms` → 立刻**停播**（`StreamingSpeaker.stop()` 打断当前播放并丢弃未播队列）
    ＋**取消当前生成**（走 `Agent.run(interrupt_check=...)`，复用既有中断语义：返回已生成部分）
    ＋**保住已经说的那半句音频**（`MicListener.take_utterance()`）→ 直接作为下一轮输入，不用「再说一遍」

**可测性（这是本文件的设计重点）**：麦克风与播放器都是**可注入接口**——
`AudioSource`（真麦克风 / 音频文件当麦克风 / 脚本化合成音频）与 `AudioSink`（真播放 / 只记录不发声）。
VAD 与切句是**纯状态机**，不碰硬件。于是「分段边界 / 打断时序 / 流式顺序」都能在 CI 里确定性跑，
不需要真麦克风、不需要出声、不需要模型。真机验收（戴耳机说话）只需要跑最后一遍。
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave

from .aec import AecEngine, ReferenceTap, ResidualSuppressor, make_aec

# ══════════════════════════════════════════════════════════════════
# 依赖检查（Phase 1 沿用）
# ══════════════════════════════════════════════════════════════════
def _check_deps() -> list[str]:
    """返回缺失的依赖提示列表（空 = 齐了）。

    用 `find_spec` 判断（**不真导入**：whisper/torch 导入一次要好几秒）。
    """
    hints = {"sounddevice": "sounddevice", "numpy": "numpy", "edge_tts": "edge-tts",
             "whisper": "openai-whisper"}
    missing = []
    for mod, pkg in hints.items():
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(f"{mod}（安装：pip install {pkg}）")
        except Exception:
            missing.append(f"{mod}（安装：pip install {pkg}）")
    return missing


# ══════════════════════════════════════════════════════════════════
# STT / TTS 引擎（可插拔，A18）
# ══════════════════════════════════════════════════════════════════
class STTEngine:
    """语音转文字。`transcribe` 接受**文件路径**或 **numpy float32 单声道数组**。"""

    name = "base"

    def transcribe(self, audio) -> str:                    # pragma: no cover - 子类实现
        raise NotImplementedError

    def transcribe_partial(self, audio) -> str:
        """流式草稿（说话中途的近似结果）。默认退回整段转写——引擎不支持流式也能用。"""
        return self.transcribe(audio)


class WhisperSTT(STTEngine):
    name = "whisper"

    def __init__(self, model: str = "base", language: str = "zh"):
        import whisper

        self.model_name = model
        self.language = language
        self.model = whisper.load_model(model)

    def transcribe(self, audio) -> str:
        kw = {"language": self.language} if self.language else {}
        # whisper 直接吃 numpy 数组（16k float32 单声道），省掉临时文件
        result = self.model.transcribe(audio, fp16=False, **kw)
        return (result.get("text") or "").strip()

    def transcribe_partial(self, audio) -> str:
        """草稿模式：贪心解码、不带上文条件，只要快。"""
        kw = {"language": self.language} if self.language else {}
        try:
            result = self.model.transcribe(audio, fp16=False, condition_on_previous_text=False,
                                           temperature=0.0, without_timestamps=True, **kw)
        except TypeError:                                  # 老版本签名差异
            result = self.model.transcribe(audio, fp16=False, **kw)
        return (result.get("text") or "").strip()


class TTSEngine:
    name = "base"

    def synthesize(self, text: str, out_path: str) -> str:  # pragma: no cover - 子类实现
        raise NotImplementedError


class EdgeTTS(TTSEngine):
    name = "edge"

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%"):
        self.voice = voice
        self.rate = rate

    def synthesize(self, text: str, out_path: str) -> str:
        import edge_tts

        async def _run():
            await edge_tts.Communicate(text, self.voice, rate=self.rate).save(out_path)

        asyncio.run(_run())
        return out_path


# ══════════════════════════════════════════════════════════════════
# 音频源（可注入）：真麦克风 / 文件当麦克风 / 脚本化合成音频
# ══════════════════════════════════════════════════════════════════
class AudioSource:
    """音频源接口：固定块长、float32 单声道、[-1,1]。"""

    samplerate = 16000
    block_ms = 200
    kind = "base"

    def __init__(self, samplerate: int = 16000, block_ms: int = 200):
        self.samplerate = samplerate
        self.block_ms = block_ms

    @property
    def block_frames(self) -> int:
        return max(1, int(self.samplerate * self.block_ms / 1000))

    def open(self) -> None:
        pass

    def read(self, frames: int):                            # pragma: no cover - 子类实现
        raise NotImplementedError

    def close(self) -> None:
        pass


class SoundDeviceSource(AudioSource):
    """真麦克风（sounddevice）。"""

    kind = "mic"

    def __init__(self, samplerate: int = 16000, block_ms: int = 200, device=None):
        super().__init__(samplerate, block_ms)
        self.device = device
        self._stream = None

    def open(self):
        if self._stream is not None:      # 幂等：AEC 提前量要在开麦后算，可能与主循环重复调用
            return self                   # 注意：守卫必须在 import 之前（CI 没装 sounddevice，否则守卫测试直接炸）
        import sounddevice as sd

        self._stream = sd.InputStream(device=self.device, samplerate=self.samplerate,
                                      channels=1, dtype="float32")
        self._stream.start()
        return self

    def read(self, frames: int):
        data, _ = self._stream.read(frames)
        return data

    @property
    def latency_s(self) -> float:
        """输入侧真实延迟（秒，PortAudio 报的）—— AEC 的参考提前量靠它推算，别拍常数。"""
        try:
            return float(getattr(self._stream, "latency", 0.0) or 0.0)
        except Exception:                                   # pragma: no cover
            return 0.0

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


class FileSource(AudioSource):
    """把音频文件当麦克风（**不用真麦克风也能跑完整链路**）。

    支持 wav / 其它格式（非 wav 走 ffmpeg 转 16k 单声道）；`realtime=True` 时按块节流，模拟真实说话速度。
    文件读完后重复（`loop=True`）或补静音，这样一轮对话里可以说多段话。
    """

    kind = "file"

    def __init__(self, path: str, samplerate: int = 16000, block_ms: int = 200,
                 realtime: bool = False, loop: bool = False, tail_silence_ms: int = 1200):
        super().__init__(samplerate, block_ms)
        self.path = path
        self.realtime = realtime
        self.loop = loop
        # 每遍之间插一段静音：否则 VAD 永远等不到「说完」（L2 真链路自测必需）
        self.tail_silence_ms = tail_silence_ms
        self._tail = 0
        self._data = None
        self._pos = 0
        # 注意：这里**不要** import numpy —— 构造不该依赖它（方法内按需导入即可），
        # 否则「无 numpy 环境下能不能构造 FileSource」都测不了

    def _load(self):
        import numpy as np

        path = self.path
        if not path.lower().endswith(".wav"):
            tmp = tempfile.mktemp(suffix=".wav")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                            "-ar", str(self.samplerate), "-ac", "1", tmp], check=True)
            path = tmp
        with wave.open(path, "rb") as wf:
            ch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype={1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)).astype(np.float32)
        if sw == 2:
            data /= 32768.0
        elif sw == 4:
            data /= 2147483648.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        if sr != self.samplerate:                      # 简单线性重采样（验证够用）
            n = int(len(data) * self.samplerate / sr)
            data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
        return data

    def open(self):
        self._data = self._load()
        self._pos = 0
        return self

    def read(self, frames: int):
        import numpy as np

        if self.realtime:
            time.sleep(frames / self.samplerate)
        else:
            time.sleep(0.005)          # 非实时也别把 CPU/GIL 打满（监听线程会拖慢主循环）
        if self._pos >= len(self._data):                   # 文件读完
            if self._tail > 0:
                self._tail -= 1
                return np.zeros((frames, 1), dtype=np.float32)
            if self.loop:                                  # 重新放一遍，但先给一段静音
                self._pos = 0
                self._tail = max(1, int(self.tail_silence_ms / max(1, self.block_ms)))
                return np.zeros((frames, 1), dtype=np.float32)
            return np.zeros((frames, 1), dtype=np.float32)
        end = self._pos + frames
        chunk = self._data[self._pos:end]
        self._pos = end
        if len(chunk) < frames:                           # 尾巴补齐静音（不跨界读下一遍）
            chunk = np.concatenate([chunk, np.zeros(frames - len(chunk), dtype=np.float32)])
        return chunk.reshape(-1, 1)


class ScriptedSource(AudioSource):
    """测试用：按脚本产生「有声 / 静音」块，纯合成，不需要任何音频文件或设备。

    script 形如 [("speech", 1.2), ("silence", 0.8), ...]（秒）；`pace` 控制每块的真实等待，
    设小一点（如 0.005）测试就跑得快，但时序关系仍然真实。
    """

    kind = "scripted"

    def __init__(self, script, samplerate: int = 16000, block_ms: int = 100,
                 freq: float = 220.0, amp: float = 0.3, pace: float = 0.005):
        super().__init__(samplerate, block_ms)
        self.script = list(script)
        self.freq = freq
        self.amp = amp
        self.pace = pace
        self._idx = 0
        self._phase = 0.0
        self._emit = []                                    # 预生成块序列
        self._pos = 0
        self._build()

    def _build(self):
        import numpy as np

        frames = self.block_frames
        for kind, secs in self.script:
            n_blocks = max(1, int(round(secs * 1000 / self.block_ms)))
            for _ in range(n_blocks):
                self._emit.append(kind)

    def open(self):
        self._pos = 0
        return self

    def read(self, frames: int):
        import numpy as np

        kind = self._emit[self._pos] if self._pos < len(self._emit) else "silence"
        self._pos += 1
        if self.pace:
            time.sleep(self.pace)
        if kind == "speech":
            t = (np.arange(frames, dtype=np.float32) + self._phase) / self.samplerate
            data = (self.amp * np.sin(2 * np.pi * self.freq * t)).astype(np.float32)
            self._phase = (self._phase + frames) % self.samplerate
        else:
            data = np.zeros(frames, dtype=np.float32)
        return data.reshape(-1, 1)

    def exhausted(self) -> bool:
        return self._pos >= len(self._emit)


# ══════════════════════════════════════════════════════════════════
# 播放汇聚（可注入）：真播放 / 只记录不发声
# ══════════════════════════════════════════════════════════════════
class PlaybackHandle:
    """一次播放的句柄：`stop()` 必须**立刻**停（打断的体感全靠它）。"""

    def stop(self) -> None:
        pass

    def wait(self, timeout=None) -> bool:
        return True

    @property
    def playing(self) -> bool:
        return False


class AudioSink:
    kind = "base"

    def play(self, path: str) -> PlaybackHandle:            # pragma: no cover - 子类实现
        raise NotImplementedError


class NullPlayback(PlaybackHandle):
    """无声播放：按音频时长「播」一遍（可被打断），供测试用。"""

    def __init__(self, seconds: float, on_stop=None):
        self.seconds = seconds
        self._stop = threading.Event()
        self._ended = threading.Event()
        self.stopped_at = None
        self.started_at = time.time()
        self._on_stop = on_stop
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        end = time.time() + self.seconds
        while not self._stop.is_set() and time.time() < end:
            time.sleep(0.01)
        self._ended.set()

    def stop(self):
        if not self._ended.is_set():
            self.stopped_at = time.time()
            self._stop.set()
            if self._on_stop:
                self._on_stop()
        else:
            self._stop.set()

    def wait(self, timeout=None) -> bool:
        return self._ended.wait(timeout)

    @property
    def playing(self):
        return not self._ended.is_set()


class NullSink(AudioSink):
    """不发声：记录每次播放与打断时刻（测试断言用）。"""

    kind = "null"

    def __init__(self, seconds_per_play: float = 0.2, audio_len=None):
        self.played = []
        self.stops = []
        self.seconds_per_play = seconds_per_play
        self.audio_len = audio_len or (lambda path: seconds_per_play)

    def play(self, path):
        self.played.append({"path": path, "at": time.time()})
        return NullPlayback(self.audio_len(path), on_stop=lambda: self.stops.append(time.time()))


def decode_audio_pcm(path: str, samplerate: int = 16000) -> "np.ndarray":
    """ffmpeg 解码成 float32 单声道 PCM —— AEC 要样本，ffplay 只有子进程拿不到。"""
    import numpy as np

    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
                           "-ar", str(int(samplerate)), "-"], capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"解码失败：{path}（ffmpeg 退出码 {proc.returncode}）")
    return np.frombuffer(proc.stdout, dtype="<f4").astype("float32")


class _SoundDevicePlayback(PlaybackHandle):
    """一段 PCM 的进程内播放；写入的同时把样本推给参考 tap（时间戳≈实际播出时刻）。"""

    def __init__(self, sink, pcm):
        self.sink = sink
        self.pcm = pcm
        self._stop = threading.Event()
        self._ended = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import sounddevice as sd
        try:
            with sd.OutputStream(samplerate=self.sink.samplerate, channels=1, dtype="float32",
                                 device=self.sink.device) as st:
                lat = 0.0
                try:
                    lat = float(st.latency or 0.0)
                except Exception:
                    lat = 0.0
                for i in range(0, len(self.pcm), self.sink.block):
                    if self._stop.is_set():
                        break
                    chunk = self.pcm[i:i + self.sink.block]
                    ref_len = st.write(chunk.reshape(-1, 1))
                    self.sink.tap.push(chunk, time.monotonic() - lat)   # 参考信号
        except Exception:
            pass
        finally:
            self._ended.set()

    def stop(self):
        self._stop.set()
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass

    def wait(self, timeout=None) -> bool:
        return self._ended.wait(timeout)

    @property
    def playing(self):
        return (not self._stop.is_set()) and (not self._ended.is_set())


class SoundDeviceSink(AudioSink):
    """进程内播放（sounddevice）+ 参考信号 tap —— **AEC 模式必须用它**。

    与 `FFplaySink` 的差别：ffplay 是外部进程，Python 拿不到它正在播的样本，就没法把自己的
    声音从麦克风里减掉；这里播放与参考共用同一份 PCM，并记录播出时刻。代价是占用音频设备。
    """

    kind = "sounddevice"

    def __init__(self, samplerate: int = 16000, tap=None, device=None, block_ms: int = 20):
        self.samplerate = int(samplerate)
        self.tap = tap if tap is not None else ReferenceTap(samplerate=samplerate)
        self.device = device
        self.block = max(1, int(self.samplerate * block_ms / 1000))

    def play(self, path):
        # ★ 每次开始播放都要**重新锚定参考时间轴**：`ReferenceTap.stream()` 的游标在第一次调用时
        #   锚定，而监听线程通常比播放先起来（差 100~300ms）→ 不重锚就整体错位、超出滤波器跨度，
        #   AEC 完全失效（2026-10-03 端到端复现挖到：残差比输入还响、VAD 仍被自己的话触发）。
        try:
            self.tap.reset_cursor()
        except Exception:                                   # pragma: no cover
            pass
        return _SoundDevicePlayback(self, decode_audio_pcm(path, self.samplerate))


class FFplaySink(AudioSink):
    """真播放：ffplay 子进程；`stop()` 直接 terminate（这就是打断的即时性来源）。"""

    kind = "ffplay"

    def __init__(self, volume: int = 100):
        self.volume = volume

    def play(self, path):
        proc = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error",
                                 "-volume", str(self.volume), path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _ProcPlayback(proc)


class _ProcPlayback(PlaybackHandle):
    def __init__(self, proc):
        self.proc = proc

    def stop(self):
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def wait(self, timeout=None) -> bool:
        try:
            self.proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    @property
    def playing(self):
        return self.proc.poll() is None


# ══════════════════════════════════════════════════════════════════
# VAD（纯状态机，可确定性测试）
# ══════════════════════════════════════════════════════════════════
SPEECH_START = "speech_start"
PARTIAL = "partial"
SPEECH_END = "speech_end"


class VadEvent:
    def __init__(self, kind, rms=0.0, elapsed=0.0):
        self.kind = kind
        self.rms = rms
        self.elapsed = elapsed

    def __repr__(self):
        return f"<VadEvent {self.kind} rms={self.rms:.4f}>"


class EnergyVAD:
    """能量阈值 VAD 状态机：说话开始 / 周期性草稿 / 说话结束。

    · `threshold`    RMS 阈值（环境底噪之上；本机实测底噪 0.005 左右，默认 0.012）
    · `silence_ms`   连续静音多久算一句话说完
    · `min_speech_ms` 短于此的「说话」当噪声丢弃（咳嗽/敲键盘）
    · `partial_ms`   说话期间每隔多久发一次草稿事件（流式转写的触发点）
    """

    def __init__(self, threshold: float = 0.012, silence_ms: int = 800, min_speech_ms: int = 300,
                 partial_ms: int = 1000):
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.partial_ms = partial_ms
        self._speaking = False
        self._speech_ms = 0
        self._silent_ms = 0
        self._since_partial_ms = 0
        self._total_ms = 0

    def reset(self):
        self._speaking = False
        self._speech_ms = self._silent_ms = self._since_partial_ms = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def feed(self, block, block_ms: int) -> list:
        """喂一块音频（numpy 数组），返回本次产生的事件列表。"""
        import numpy as np

        arr = np.asarray(block, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
        self._total_ms += block_ms
        events = []
        if rms >= self.threshold:
            self._speech_ms += block_ms
            self._silent_ms = 0
            if not self._speaking:
                self._speaking = True
                self._since_partial_ms = 0
                events.append(VadEvent(SPEECH_START, rms, self._total_ms))
            else:
                self._since_partial_ms += block_ms
                if self._since_partial_ms >= self.partial_ms:
                    self._since_partial_ms = 0
                    events.append(VadEvent(PARTIAL, rms, self._total_ms))
        else:
            self._silent_ms += block_ms
            if self._speaking:
                self._since_partial_ms += block_ms
                if self._silent_ms >= self.silence_ms:
                    self._speaking = False
                    if self._speech_ms >= self.min_speech_ms:
                        # elapsed 带上「这句话说了多久」（毫秒）——调用方用它判断音频够不够长
                        events.append(VadEvent(SPEECH_END, rms, self._speech_ms))
                    else:
                        self._speech_ms = 0          # 太短，当噪声丢掉
        return events


# ══════════════════════════════════════════════════════════════════
# 说话期间听麦克风（打断检测 + 保住已说半句）
# ══════════════════════════════════════════════════════════════════
class KeyHold:
    """「按住说话」的按键状态源 —— **可注入**，所以 PTT 逻辑能在 CI 里确定性测（塞个假的）。

    Windows 用 `GetAsyncKeyState` 轮询：它能回答「现在按着吗」，而 `msvcrt` 只给按下事件、
    判断不了长按。非 Windows / 无 GUI 环境返回 False（PTT 主要用于本地 Windows）。
    """

    def __init__(self, vk: int = 0x20):           # 0x20 = 空格
        self.vk = vk

    def held(self) -> bool:
        try:
            import ctypes
            return bool(ctypes.windll.user32.GetAsyncKeyState(self.vk) & 0x8000)
        except Exception:
            return False


class MicListener:
    """后台线程持续读麦克风：跑 VAD，攒当前这句话的音频，并提供「有人在说话」的即时判断。

    打断判定：连续有声 ≥ `barge_ms` 就置 `barge_in=True`（哪怕 forge 正在播放/生成）。
    `take_utterance()` 取出「已经说出来的那半句」的音频 —— 抢话后不用让用户再说一遍。
    """

    def __init__(self, source: AudioSource, vad: EnergyVAD = None, barge_ms: int = 300,
                 aec: "AecEngine" = None, ref_tap=None, aec_lead_ms: int = 120,
                 suppressor: "ResidualSuppressor" = None):
        self.source = source
        self.vad = vad or EnergyVAD()
        self.barge_ms = barge_ms
        # AEC（回声消除）：外放时把「自己播出去的声音」从麦克风里减掉，避免自激——
        # 这样它说话时仍能听见你插话（半双工/PTT 都做不到这点）。
        self.aec = aec
        self.ref_tap = ref_tap
        self.aec_lead_ms = int(aec_lead_ms)
        # 残余回声抑制（AEC 之后那一步）：喇叭→麦克风非线性时 AEC 消不掉，靠它压住（见 aec.py）
        self.suppressor = suppressor
        self.events = queue.Queue()
        self._frames = []                 # 当前这句话的音频
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.barge_in = False
        self.speech_ms_run = 0            # 连续有声时长（判定抢话用）
        self.closed = False
        self._prev_block = None           # 上一块（SPEECH_START 时当 pre-roll 用）
        self._utterance = []              # SPEECH_END 时冻结下来的「这一句」（避免缓冲区继续被灌）
        self._muted = False               # 闭麦：外放时喇叭里的 TTS 会被自己听见（半双工/PTT 用）

    # ---- 生命周期 ----
    def start(self):
        self.source.open()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    block = self.source.read(self.source.block_frames)
                except Exception as e:                       # 源结束/设备异常
                    self.events.put(VadEvent(SPEECH_END))
                    self.closed = True
                    break
                with self._lock:
                    if self._muted:                          # 闭麦：这一块当没听见
                        continue
                if self.aec is not None and self.ref_tap is not None:
                    block = self._cancel_echo(block)
                for ev in self.vad.feed(block, self.source.block_ms):
                    if ev.kind == SPEECH_START:
                        self.speech_ms_run = 0
                        with self._lock:
                            # 只留发声起点之后的音频（前面攒的静音丢掉），
                            # 留一块 pre-roll 防切掉字头
                            self._frames = ([self._prev_block] if self._prev_block is not None else [])
                    elif ev.kind == SPEECH_END:
                        # **把这一句冻结**：否则源读得快（文件源）或主循环忙时，
                        # 缓冲区会继续被静音灌满 → 转写拿到几十秒静音 → 识别为空（2026-09-29 实测踩到）
                        with self._lock:
                            self._utterance, self._frames = self._frames, []
                    self.events.put(ev)
                with self._lock:
                    self._frames.append(block)
                    self._prev_block = block
                    if len(self._frames) > 400:              # 防空转吃内存（约 80s @200ms）
                        self._frames = self._frames[-200:]
                # 连续有声时长（抢话判定，独立于 VAD 的状态）
                if ev_rms_high(block, self.vad.threshold):
                    self.speech_ms_run += self.source.block_ms
                    if self.speech_ms_run >= self.barge_ms:
                        self.barge_in = True
                else:
                    self.speech_ms_run = 0
        finally:
            pass

    def _cancel_echo(self, block):
        """把「刚播出去的声音」从这一块麦克风里减掉（AEC），再交给 VAD/抢话判定。

        参考段用 `lead_s` 粗对齐（设备缓冲 + 声学延迟）；滤波器长度本身覆盖延迟不确定性。
        """
        import numpy as np

        shape = getattr(block, "shape", None)
        blk = np.asarray(block, dtype=np.float32).reshape(-1)
        # 用 stream()（一次锚定后按样本计数推进）：每块都用墙钟会让滤波器学糊，见 ReferenceTap.stream
        ref = self.ref_tap.stream(blk.size, self.aec_lead_ms / 1000.0)
        out = self.aec.process(blk, ref)
        if self.suppressor is not None:
            out = self.suppressor.process(out, ref)   # AEC 之后再来一道：压住非线性残余回声
        if not self.aec.converged():
            # 冷启动收敛期：残差里还有回声，把它当静音 —— 否则会「自己触发自己」
            # （真实产品同样有这段保护；代价是整场开头 ~300ms 听不见，用户说话一般不会正好卡在这）
            return np.zeros_like(blk).reshape(shape) if shape else np.zeros_like(blk)
        return out.reshape(shape) if shape else out

    def set_muted(self, muted: bool):
        """闭麦 / 开麦。闭麦期间读到的音频**直接丢**：不喂 VAD、不攒缓冲、不判抢话。

        为什么闭麦而不是停采集：设备/流重开有驱动抖动与延迟；而「外放时喇叭里的 TTS
        被自己的麦克风听见」纯粹是多余输入，丢掉就行（2026-09-29：用户问不用耳机怎么办）。
        """
        with self._lock:
            self._muted = bool(muted)
        if muted:
            self.clear()
            self.vad.reset()
            self.speech_ms_run = 0
            self.barge_in = False

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.source.close()
        except Exception:
            pass

    # ---- 查询 / 取值 ----
    def next_event(self, timeout=0.1):
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_events(self):
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def take_utterance(self):
        """取走「这一句」的音频（优先冻结句；抢话时给正在说的那半句）。"""
        import numpy as np

        with self._lock:
            frames = self._utterance or self._frames
            self._utterance, self._frames = [], []
        if not frames:
            return None
        return np.concatenate([f.reshape(-1) for f in frames]).astype(np.float32)

    def peek_audio(self):
        """看一眼当前已采集的音频（**不清空**，草稿转写用）。"""
        import numpy as np

        with self._lock:
            frames = list(self._utterance or self._frames)
        if not frames:
            return None
        return np.concatenate([f.reshape(-1) for f in frames]).astype(np.float32)

    def peek_seconds(self) -> float:
        with self._lock:
            n = sum(len(f) for f in self._frames)
        return n / float(self.source.samplerate)

    def clear(self):
        self._utterance = []
        self.take_utterance()
        self.barge_in = False
        self.speech_ms_run = 0
        self.vad.reset()
        self.drain_events()

    # 供 Agent 的 interrupt_check 直接调用（同步、非阻塞）
    def __call__(self) -> bool:
        return self.barge_in


def ev_rms_high(block, threshold: float) -> bool:
    import numpy as np

    arr = np.asarray(block, dtype=np.float32).reshape(-1)
    return bool(arr.size and float(np.sqrt(np.mean(arr ** 2))) >= threshold)


# ══════════════════════════════════════════════════════════════════
# 句子切分（流式）—— 让 TTS 能「说一句合成一句」
# ══════════════════════════════════════════════════════════════════
class SentenceSplitter:
    """边收模型增量边切句。

    · 遇到句末标点（。！？!?…；;\\n）且缓冲 ≥ `min_chars` 就吐一句
    · 缓冲到 `max_chars` 还没句末就强制切（优先在逗号/顿号处切，实在没有就硬切）
    · 顺手把 Markdown 标记清掉（TTS 念「星号星号」很出戏）
    """

    PUNCT = "。！？!?…；;"
    SOFT = "，,、）)】」》"

    def __init__(self, min_chars: int = 8, max_chars: int = 120):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.buf = ""

    @staticmethod
    def clean_for_tts(text: str) -> str:
        """去掉不该念出来的东西：Markdown 标记、代码围栏、URL、多余空白。"""
        import re

        t = re.sub(r"```.*?```", "（代码略）", text, flags=re.S)
        t = re.sub(r"`([^`]*)`", r"\1", t)
        t = re.sub(r"\*\*([^*]*)\*\*", r"\1", t)
        t = re.sub(r"(^|\s)[*#>-]{1,6}\s*", r"\1", t)
        t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
        t = re.sub(r"https?://\S+", "链接", t)
        t = re.sub(r"[ \t]+", " ", t)
        return t.strip()

    def feed(self, delta: str) -> list:
        if not delta:
            return []
        self.buf += delta
        out = []
        while True:
            piece = self._pop_ready()
            if piece is None:
                break
            cleaned = self.clean_for_tts(piece)
            if cleaned:
                out.append(cleaned)
        return out

    def _pop_ready(self):
        buf = self.buf
        # 1) 句末标点 + 够长
        for i, ch in enumerate(buf):
            if ch in self.PUNCT and i + 1 >= self.min_chars:
                self.buf = buf[i + 1:]
                return buf[:i + 1]
        # 2) 超长强制切（优先软标点）
        if len(buf) >= self.max_chars:
            cut = max(buf.rfind(c, 0, self.max_chars) for c in self.SOFT)
            cut = cut if cut >= self.min_chars else self.max_chars
            self.buf = buf[cut:]
            return buf[:cut]
        return None

    def flush(self) -> list:
        rest, self.buf = self.buf, ""
        cleaned = self.clean_for_tts(rest)
        return [cleaned] if cleaned else []


# ══════════════════════════════════════════════════════════════════
# 流式播放器（句子级流水线 + 立即打断）
# ══════════════════════════════════════════════════════════════════
class StreamingSpeaker:
    """一句合成完就播，同时下一句在合成；`stop()` 立即打断当前播放并丢弃未播队列。"""

    def __init__(self, tts: TTSEngine, sink: AudioSink, tmpdir: str = None, on_error=None):
        self.tts = tts
        self.sink = sink
        self.tmpdir = tmpdir
        self.on_error = on_error
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._current = None
        self._lock = threading.Lock()
        self._epoch = 0            # 轮次代号：打断后旧代号的句子一律作废（见 start()/_retired）
        self.said = []
        self.spoken_chars = 0
        self.finished = threading.Event()
        self.finished.set()

    def start(self):
        with self._lock:
            self._epoch += 1                   # 新一轮 = 新代号 → 上一轮残留/合成中的句子作废
            my_epoch = self._epoch
        self._stop.clear()
        # 清掉上一轮遗留（**包括 stop() 可能留下的哨兵**）：
        # 否则新一轮的工作线程一启动就吃到哨兵立刻退出 → 整段回答无声
        # （2026-09-29 PTT 用例抓到：第一轮之前按一下键就会触发）
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self.finished.clear()
        self._thread = threading.Thread(target=self._run, args=(my_epoch,), daemon=True)
        self._thread.start()
        return self

    def _retired(self, my_epoch: int) -> bool:
        """本工作线程是否该退休：被打断（`_stop`）或被新一轮取代（代号变了）。

        为什么必须有代号：`stop()` 只设 `_stop`，而下一轮 `start()` 会**清掉**它 —— 于是「合成
        期间被打断」的那句合成完回来时看不到停止标记，就照播了（2026-10-03 老大听到的「把打断
        前的回答也念了」）。代号一变旧句一律作废，不再依赖那个会被清掉的标记。
        """
        return self._stop.is_set() or my_epoch != self._epoch

    def _run(self, my_epoch: int):
        while not self._retired(my_epoch):
            try:
                text = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            if text is None or self._retired(my_epoch):
                break
            try:
                path = tempfile.mktemp(suffix=".mp3", dir=self.tmpdir)
                self.tts.synthesize(text, path)
                if self._retired(my_epoch):
                    self._cleanup(path)      # 合成期间被打断 → 丢弃这一句，别漏播
                    break
                handle = self.sink.play(path)
                with self._lock:
                    self._current = handle
                self.said.append(text)
                self.spoken_chars += len(text)
                # 等播完，但被打断/被新一轮取代时立刻退出
                while handle.playing and not self._retired(my_epoch):
                    time.sleep(0.01)
                if self._retired(my_epoch):
                    handle.stop()
                self._cleanup(path)
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
            finally:
                with self._lock:
                    self._current = None
        self.finished.set()

    @staticmethod
    def _cleanup(path):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass

    def say(self, text: str):
        """入队一句（同步、非阻塞：模型增量回调里直接调）。"""
        if text and not self._stop.is_set():
            self._q.put(text)

    def stop(self):
        """立即打断：停当前播放 + 清空未播队列。"""
        self._stop.set()
        with self._lock:
            handle = self._current
        if handle is not None:
            handle.stop()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if self._thread is not None and self._thread.is_alive():
            self._q.put(None)          # 哨兵只给活着的线程（线程没起来时塞进去会毒到下一轮）

    def finish(self, timeout=60.0) -> bool:
        """等队列排空且当前这句播完（**不是等线程退出**——线程要等 close() 的哨兵）。"""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                cur = self._current
            if self._q.empty() and (cur is None or not cur.playing):
                return True
            time.sleep(0.01)
        return False

    def close(self):
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ══════════════════════════════════════════════════════════════════
# Phase 1（保留）：录音到静默 / 播放 / 整句级循环
# ══════════════════════════════════════════════════════════════════
def record_until_silence(out_path: str, samplerate: int = 16000, silence_seconds: float = 0.8,
                         max_seconds: float = 30.0) -> float:
    """录音直到静默（能量阈值判定）或超时，返回实际录音时长（秒）。"""
    import numpy as np
    import sounddevice as sd

    block = int(samplerate * 0.2)
    threshold = 0.012
    chunks, silent_blocks = [], 0
    max_blocks = int(max_seconds / 0.2)
    required_silent = max(1, int(silence_seconds / 0.2))
    print("🎙 请说话…（说完静默自动结束，最多 %.0fs）" % max_seconds)
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
        started = False
        for _ in range(max_blocks):
            data, _ = stream.read(block)
            rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
            if rms >= threshold:
                chunks.append(data.copy())
                silent_blocks = 0
                started = True
            elif started:
                chunks.append(data.copy())
                silent_blocks += 1
                if silent_blocks >= required_silent:
                    break
    if not chunks:
        return 0.0
    audio = np.concatenate(chunks)
    write_wav(out_path, audio, samplerate)
    return len(audio) / samplerate


def write_wav(path: str, audio, samplerate: int = 16000) -> str:
    """float32 [-1,1] 单声道数组 → wav 文件。"""
    import numpy as np

    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((arr * 32767).astype(np.int16).tobytes())
    return path


def play_audio(path: str) -> None:
    """播放音频（优先 ffplay）。"""
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path], check=False)
    except FileNotFoundError:                              # pragma: no cover
        import sys

        if sys.platform == "darwin":
            subprocess.run(["afplay", path], check=False)
        elif sys.platform.startswith("win"):
            subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
                           check=False)


# 语音默认值：**配置文件可以改默认，命令行开关仍然最优先**（优先级：命令行 > config > 这里）。
# 这样在 `config/models.yaml` 里写一次 `voice: {aec: nlms}`，之后裸跑 `forge --voice` 就是
# 「外放免耳机、不用按键、它说话时也能插话」——不用每次手打开关。
# （2026-10-03 老大问「为啥不能用 forge --voice」，就是不想每次加 --aec。）
VOICE_DEFAULTS = {
    "audio_source": "mic",      # mic | file:<wav>（文件源用于自测/回归）
    "sink": "speaker",          # speaker（AEC 模式会自动换成进程内播放）| null
    "rounds": 0,                # 0 = 不限轮数
    "barge_ms": 300,            # 抢话阈值：连续有声多久算插话
    "stt_model": "base",        # whisper 模型（small 中文更准）
    "aec": None,                # None/None = 不开；"nlms"（默认引擎）/ "pyaec" / "none"
    "aec_lead_ms": 0,           # 0 = 自动按设备输入延迟推算
    "half_duplex": False,       # 播放期间闭麦（免耳机但播放时不能插话）
    "ptt": False,               # 按住空格说话
}


def voice_defaults(cfg: dict = None) -> dict:
    """从配置里取语音默认值：**只认已知键**（避免配置里塞错东西污染参数）。

    只有 `cfg["voice"]` 是 dict 且键在白名单里才生效；其余一律用内置默认。
    """
    out = dict(VOICE_DEFAULTS)
    section = (cfg or {}).get("voice")
    if not isinstance(section, dict):
        return out
    for k in VOICE_DEFAULTS:
        if k in section and section[k] is not None:
            out[k] = section[k]
    return out


def estimate_aec_lead_ms(source, margin_ms: int = 15, fallback_ms: int = 120,
                         lo: int = 40, hi: int = 180) -> int:
    """AEC 参考提前量 = **输入侧延迟 + 余量**（输出侧已在推参考时按流延迟补偿过）。

    为什么不能拍常数：本机实测输入延迟就有 90ms（Realtek 默认），拍 120 差 30ms、拍 0 差 90ms。
    差得越多，自适应滤波器就得把越多抽头花在「对齐」而不是「消回声」上（长度有限，会拉低 ERLE）。
    上下限 40~180ms：太小盖不住设备缓冲，太大则超出常规「设备缓冲 + 声学延迟」量级（滤波器按 200ms 设计）。
    """
    lat = getattr(source, "latency_s", None)
    if not lat:
        return fallback_ms                                     # 无延迟信息（文件源/假源）→ 保守值
    return int(max(lo, min(hi, round(float(lat) * 1000) + margin_ms)))


async def voice_loop(agent, stt: STTEngine, tts: TTSEngine, keep_alive: bool = True,
                     max_rounds: int = 0) -> None:
    """Phase 1：整句级语音对话（说一句 → 答一句）。保留给「只要最简链路」的场景。"""
    print("🔊 语音模式已开启（Phase 1 整句级；说「退出」/「exit」结束）")
    rounds = 0
    while True:
        rounds += 1
        if max_rounds and rounds > max_rounds:
            break
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "input.wav")
            secs = record_until_silence(wav)
            if secs < 0.3:
                print("（没听清，再说一次）")
                continue
            print(f"⏳ 识别中…（{secs:.1f}s 音频）")
            text = stt.transcribe(wav).strip()
            if not text:
                print("（识别为空，再说一次）")
                continue
            print(f"🗣 你说：{text}")
            if is_exit(text):
                print("👋 语音模式结束")
                break
            answer = await agent.run(text)
            print(f"🔨 forge：{answer}")
            out_mp3 = os.path.join(tmp, "reply.mp3")
            tts.synthesize(answer, out_mp3)
            play_audio(out_mp3)
        if not keep_alive:
            break


EXIT_WORDS = ("退出", "exit", "结束", "再见", "不聊了", "拜拜")


def is_exit(text: str) -> bool:
    t = (text or "").strip().lower().rstrip("。.!！,，")
    return t in EXIT_WORDS


# ══════════════════════════════════════════════════════════════════
# Phase 2/3：流式 + 打断
# ══════════════════════════════════════════════════════════════════
def await_playback(speaker: "StreamingSpeaker", listener: "MicListener", answer: str,
                   cap_seconds: float = None, extra_check=None) -> bool:
    """等这句话播完，**同时盯着抢话**——用户听的时候插话，必须立刻停播。

    `extra_check` 是额外的「立刻打断」来源（PTT 模式传按键状态：**按下即打断**，
    否则要等当前这句播完才轮到按键轮询）。
    返回 True 表示播放期间被打断（调用方应 `speaker.stop()` 并接着处理新输入）。
    """
    if listener.barge_in:
        return True
    end = time.time() + (cap_seconds if cap_seconds else max(10.0, len(answer or "") / 8))
    while time.time() < end:
        if listener.barge_in:
            return True
        if extra_check is not None and extra_check():
            return True
        if speaker.finish(timeout=0.05):          # 队列排空且当前句播完
            return False
    return False


class VoiceEvent:
    """给外部（CLI/测试）观察用的事件。"""

    def __init__(self, kind, **kw):
        self.kind = kind
        self.data = kw
        self.t = time.time()

    def __repr__(self):
        return f"<VoiceEvent {self.kind} {self.data}>"


async def streaming_voice_loop(agent, stt: STTEngine, tts: TTSEngine,
                               source: AudioSource = None, sink: AudioSink = None,
                               listener: MicListener = None, splitter: SentenceSplitter = None,
                               keep_alive: bool = True, max_rounds: int = 0,
                               on_event=None, barge_ms: int = 300, timeout_s: float = 60.0,
                               half_duplex: bool = False, guard_ms: int = 300,
                               ptt=None, aec=None, ref_tap=None, aec_lead_ms: int = 120,
                               suppressor=None) -> dict:
    """Phase 2/3 主循环：流式转写 → 流式应答（句级合成播放）→ 播放期间可抢话。

    **不用耳机也能用的两种模式**（默认全双工 = 假设你戴了耳机）：
    · `half_duplex=True`：**播放期间闭麦**（外放时喇叭里的 TTS 会被自己的麦克风听见 →
      自激：它把自己的话当新指令）。代价是播放时不能插话；**生成期间仍可插话**
      （那时喇叭还没出声，不存在回声）。`guard_ms` = 停播后等喇叭余音散掉的静默期。
    · `ptt=<KeyHold>`：**按住说话** —— 不按不采集；按下 = 打断正在说的 + 开麦；松开 = 提交这一句。
      完全不用耳机也能随时插话；句子边界是「松手」而不是 VAD。

    返回统计 dict（轮数 / 打断次数 / 事件数），测试与 CLI 都用它判断。
    """
    stats = {"rounds": 0, "barge_ins": 0, "events": 0, "transcripts": [], "answers": []}
    emits = []

    def emit(kind, **kw):
        ev = VoiceEvent(kind, **kw)
        emits.append(ev)
        stats["events"] += 1
        if on_event:
            try:
                on_event(ev)
            except Exception:
                pass
        return ev

    if listener is None:
        if source is None:
            raise ValueError("需要 source 或 listener 之一（真麦克风用 SoundDeviceSource）")
        listener = MicListener(source, EnergyVAD(), barge_ms=barge_ms, suppressor=suppressor,
                               aec=aec, ref_tap=ref_tap, aec_lead_ms=aec_lead_ms)
    listener.start()
    splitter = splitter or SentenceSplitter()
    speaker = StreamingSpeaker(tts, sink or NullSink())
    ptt_down = False
    if ptt is not None:                     # PTT：不按不采集（外放也不会听自己）
        listener.set_muted(True)

    carry = {}          # 半双工：闭麦会清掉缓冲，抢话时先把用户那半句接住（见下）

    async def _turn(text):
        """一轮应答：边生成边切句边播；返回 (answer, 是否被抢话)。

        抽出来是因为「抢话后用户那半句」要走**同一套**流程（曾经复制粘贴过一份，DRY 掉了）。
        """
        stats["rounds"] += 1
        splitter.buf = ""
        listener.clear()
        listener.barge_in = False
        speaker.start()

        def _on_delta(chunk):
            for sentence in splitter.feed(chunk):
                emit("sentence", text=sentence)
                speaker.say(sentence)

        try:
            answer = await agent.run(text, on_delta=_on_delta, interrupt_check=listener)
        except TypeError:                      # 兼容旧 Agent（无 on_delta / interrupt_check）
            answer = await agent.run(text)
            _on_delta(answer)
        for sentence in splitter.flush():
            emit("sentence", text=sentence)
            speaker.say(sentence)

        # 生成期没被抢话 → 边播边继续盯（**播放期间插话才是最常发生的**）
        hit = listener.barge_in             # **先读再闭麦**：闭麦会清掉抢话标记与缓冲
        if hit:
            carry["pending"] = listener.take_utterance()   # 用户那半句先接住，别被闭麦清掉
        if half_duplex:
            listener.set_muted(True)        # 外放：喇叭在响，别把自己的声音当用户
        hit = hit or await_playback(speaker, listener, answer,
                                    extra_check=(ptt.held if ptt is not None else None))
        if half_duplex:
            if guard_ms:
                await asyncio.sleep(guard_ms / 1000.0)      # 等喇叭余音散掉再开麦
            listener.set_muted(False)
        if hit:
            stats["barge_ins"] += 1
            speaker.stop()
            emit("barge_in", partial_answer=answer)
        else:
            emit("answer_done", answer=answer)
        stats["answers"].append(answer)
        return answer, hit

    async def _wait_speech_end(timeout_s: float = 10.0):
        """等用户把这句话说完（收到 SPEECH_END）再返回音频；超时则有多少算多少。

        为什么必须等：抢话判定只要 `barge_ms`（默认 300ms）就触发，**那一刻用户通常才说了开头
        几个字**。原来立刻转写 → 只识别到「前半句」（2026-10-03 老大实测反馈）。等他停下再转，
        「插话 = 说一句完整的新指令」才成立。用户已停（VAD 800ms 静默）时这里几乎立刻返回，
        所以不为难人；10s 只是「一直在说」的安全网。
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for ev in listener.drain_events():
                if ev.kind == SPEECH_END:
                    return listener.take_utterance()
            await asyncio.sleep(0.05)
        return listener.take_utterance()

    async def _respond(text):
        """一轮输入 → 一轮应答（含抢话后接着说的那一轮）。返回 False = 该退出了。"""
        stats["transcripts"].append(text)
        emit("user", text=text)
        if is_exit(text):
            emit("exit")
            return False
        _answer, hit = await _turn(text)
        # 抢话后：用户那半句已经被 listener 收着 → 直接进下一轮（不用再说一遍）
        # PTT 模式除外：下一句由「按键边界」定义，不该把残留音频当输入
        if hit and ptt is None:
            pending = carry.pop("pending", None)
            if pending is None:
                pending = listener.take_utterance()
            # 抢话那一刻抓到的只是开头（barge_ms 就那么多）→ **等用户说完**再一起转写，
            # 否则就是「只识别了前半句」（2026-10-03 实测）。两段音频按时间顺序拼起来。
            rest = await _wait_speech_end()
            if rest is not None and len(rest):
                import numpy as np
                head = np.asarray(pending, dtype=np.float32).reshape(-1) if (pending is not None and len(pending)) else None
                pending = np.concatenate([head, np.asarray(rest, dtype=np.float32).reshape(-1)]) if head is not None \
                    else np.asarray(rest, dtype=np.float32).reshape(-1)
            seconds = len(pending) / listener.source.samplerate if pending is not None else 0.0
            emit("barge_pending", seconds=round(seconds, 2))
            if pending is not None and len(pending) > 0:
                text2 = stt.transcribe(pending).strip()
                if text2:
                    stats["transcripts"].append(text2)
                    emit("user", text=text2)
                    await _turn(text2)
        speaker.close()
        return True

    try:
        emit("ready", source=getattr(listener.source, "kind", "?"))
        last_activity = time.time()
        while True:
            if max_rounds and stats["rounds"] >= max_rounds:
                break
            if timeout_s and (time.time() - last_activity) > timeout_s:
                emit("timeout")
                break
            # ── 按住说话（PTT）：不按不采集；按下=打断+开麦；松开=提交这一句 ──
            if ptt is not None:
                held = bool(ptt.held())
                if held and not ptt_down:
                    ptt_down = True
                    last_activity = time.time()
                    listener.clear()
                    listener.set_muted(False)
                    speaker.stop()                      # 按下即打断正在说的（这就是插话）
                    emit("ptt_down")
                elif ptt_down and not held:
                    ptt_down = False
                    audio = listener.take_utterance()   # **先取再闭麦**：闭麦会清掉缓冲
                    listener.set_muted(True)
                    secs = (len(audio) / listener.source.samplerate) if audio is not None else 0.0
                    last_activity = time.time()
                    emit("ptt_up", seconds=round(secs, 2))
                    if audio is None or secs < 0.2:
                        emit("too_short", speech_ms=int(secs * 1000))
                    else:
                        text = stt.transcribe(audio).strip()
                        if not text:
                            emit("empty")
                        elif not await _respond(text):
                            break
                elif ptt_down:
                    last_activity = time.time()         # 一直按着不算空闲（别在说话时超时退出）
                await asyncio.sleep(0.03)               # 30ms 轮询：手感够，不烧 CPU
                continue

            ev = listener.next_event(timeout=0.1)
            if ev is None:
                continue
            if ev.kind == SPEECH_START:
                emit("speech_start")
            elif ev.kind == PARTIAL:
                audio = listener.peek_audio()
                if audio is not None and len(audio) > 0:
                    draft = stt.transcribe_partial(audio)
                    if draft:
                        emit("partial", text=draft)
            elif ev.kind == SPEECH_END:
                if ptt is not None:
                    continue                    # PTT 模式：句子边界是「松手」，不是 VAD
                audio = listener.take_utterance()
                if (ev.elapsed or 0) < 200 or audio is None or len(audio) < int(0.1 * listener.source.samplerate):
                    emit("too_short", speech_ms=ev.elapsed)
                    continue
                text = stt.transcribe(audio).strip()
                if not text:
                    emit("empty")
                    continue
                last_activity = time.time()
                if not await _respond(text):
                    break
            if not keep_alive and stats["rounds"] >= 1:
                break
    finally:
        if aec is not None:
            stats["aec"] = aec.stats()
        try:
            speaker.stop()
            speaker.close()
        except Exception:
            pass
        listener.stop()
        emit("closed", **{k: v for k, v in stats.items() if isinstance(v, int)})
    return stats


def run_voice(agent, audio_source: str = "mic", rounds: int = 0, sink: str = "speaker",
              barge_ms: int = 300, stt: STTEngine = None, tts: TTSEngine = None,
              stt_model: str = "base", stream: bool = True, file_loop: bool = False,
              half_duplex: bool = False, ptt: bool = False,
              aec: str = None, aec_lead_ms: int = 120) -> dict:
    """命令行入口：`forge --voice [--audio-source mic|file:PATH] [--voice-rounds N] [--voice-sink null]`。

    `--audio-source file:xxx.wav` = **不用麦克风也能跑完整语音链路**（L2 自测/回归用）。
    `--half-duplex` = 外放不用耳机（播放期间闭麦）；`--ptt` = 按住空格说话。
    `--aec [nlms|pyaec]` = **回声消除**（真免手：不用闭麦也不用按键，它说话时你插话照样听得见）。
    """
    missing = [m for m in _check_deps() if m not in ("sounddevice",)] if audio_source.startswith("file:") else _check_deps()
    if missing:
        print("❌ 语音模式需要以下依赖（装一次即可）：")
        for m in missing:
            print(f"   · {m}")
        print('   安装示例：pip install "handcraft-agent[voice]"')
        return {"error": "missing_deps", "missing": missing}

    stt = stt or WhisperSTT(model=stt_model)
    tts = tts or EdgeTTS()
    if audio_source.startswith("file:"):
        # 默认**不循环**：循环会把同一句话再听一遍，被当成「用户插话」（自测时常踩）
        src = FileSource(audio_source.split(":", 1)[1], realtime=False, loop=file_loop)
    else:
        src = SoundDeviceSource()
    if not stream:                                  # Phase 1（整句级，保留的老链路）
        print(f"🔊 语音模式（Phase 1 整句级）· 音频源 {audio_source}")
        asyncio.run(voice_loop(agent, stt, tts, keep_alive=True, max_rounds=rounds))
        return {"mode": "phase1"}
    aec_engine = None
    ref_tap = None
    if aec and str(aec).lower() not in ("none", "off", "0"):
        # AEC 需要「正在播的音频」当参考信号 ⇒ 播放必须在进程内（ffplay 是外部进程，拿不到样本）
        aec_engine = make_aec(aec, samplerate=16000)
        ref_tap = ReferenceTap(samplerate=16000)
        suppressor = ResidualSuppressor()      # AEC 之后那一步：非线性路径也能压住残余回声
        if sink == "null":
            sink_obj = NullSink()
            print(f"🔊 回声消除（AEC={aec_engine.name}）已开，但播放是 null（无声）→ 没有参考信号，等价直通")
        else:
            sink_obj = SoundDeviceSink(samplerate=16000, tap=ref_tap)
            if not aec_lead_ms and hasattr(src, "open") and not getattr(src, "latency_s", 0):
                try:
                    src.open()            # 先开麦才有真实输入延迟可读（2026-10-03：不开就读到 0 → 悄悄回落）
                except Exception:
                    pass
            lead_ms = int(aec_lead_ms) if aec_lead_ms else estimate_aec_lead_ms(src)
            in_ms = round((getattr(src, "latency_s", 0) or 0) * 1000)
            who = "指定" if aec_lead_ms else f"自动·输入延迟{in_ms}ms+余量"
            print(f"🔊 回声消除（AEC={aec_engine.name}）· 进程内播放（sounddevice）· 参考提前量 {lead_ms}ms（{who}）")
            print("   （外放时它说话你也能插话：麦克风里的回声会被减掉，不用闭麦也不用按键）")
    else:
        sink_obj = NullSink() if sink == "null" else FFplaySink()
    if ptt:
        mode = "按住空格说话（PTT）"
    elif half_duplex:
        mode = "半双工（播放时闭麦）"
    elif aec_engine is not None:
        mode = "全双工 + AEC 回声消除（外放免耳机，不用按键）"
    else:
        mode = "全双工（建议戴耳机）"
    print(f"🔊 语音模式（Phase 2/3 流式 + 打断）· 音频源 {audio_source} · 播放 {sink} · "
          f"抢话阈值 {barge_ms}ms · {mode}")
    if ptt:
        print("   （不按不采集；**按住空格**说话、松开提交；按下即打断它正在说的。说「退出」结束）")
    elif half_duplex:
        print("   （外放模式：它说话时不听麦，防自激；它思考时插话仍可打断。说「退出」结束）")
    elif aec_engine is not None:
        print("   （说「退出」/「exit」结束；**它说话时你直接开口就能插话**。消得不够就调 --aec-lead-ms 或降音量）")
    else:
        print("   （说「退出」/「exit」结束；说话时可直接插话打断；外放请加 --half-duplex / --ptt / --aec）")
    return asyncio.run(streaming_voice_loop(agent, stt, tts, source=src, sink=sink_obj,
                                             max_rounds=rounds, barge_ms=barge_ms,
                                             half_duplex=half_duplex,
                                             ptt=KeyHold() if ptt else None,
                                             aec=aec_engine, ref_tap=ref_tap, suppressor=suppressor,
                                             aec_lead_ms=aec_lead_ms,
                                             on_event=lambda ev: print(f"   · {ev.kind} {ev.data or ''}",
                                                                       flush=True)))
