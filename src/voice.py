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
        import sounddevice as sd

        self._stream = sd.InputStream(device=self.device, samplerate=self.samplerate,
                                      channels=1, dtype="float32")
        self._stream.start()
        return self

    def read(self, frames: int):
        data, _ = self._stream.read(frames)
        return data

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
class MicListener:
    """后台线程持续读麦克风：跑 VAD，攒当前这句话的音频，并提供「有人在说话」的即时判断。

    打断判定：连续有声 ≥ `barge_ms` 就置 `barge_in=True`（哪怕 forge 正在播放/生成）。
    `take_utterance()` 取出「已经说出来的那半句」的音频 —— 抢话后不用让用户再说一遍。
    """

    def __init__(self, source: AudioSource, vad: EnergyVAD = None, barge_ms: int = 300):
        self.source = source
        self.vad = vad or EnergyVAD()
        self.barge_ms = barge_ms
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
        self.said = []
        self.spoken_chars = 0
        self.finished = threading.Event()
        self.finished.set()

    def start(self):
        self._stop.clear()
        self.finished.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while True:
            try:
                text = self._q.get(timeout=0.05)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if text is None:
                break
            if self._stop.is_set():
                break
            try:
                path = tempfile.mktemp(suffix=".mp3", dir=self.tmpdir)
                self.tts.synthesize(text, path)
                if self._stop.is_set():
                    self._cleanup(path)
                    break
                handle = self.sink.play(path)
                with self._lock:
                    self._current = handle
                self.said.append(text)
                self.spoken_chars += len(text)
                # 等播完，但 stop() 时立刻退出
                while handle.playing and not self._stop.is_set():
                    time.sleep(0.01)
                if self._stop.is_set():
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
        self._q.put(None)

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
                   cap_seconds: float = None) -> bool:
    """等这句话播完，**同时盯着抢话**——用户听的时候插话，必须立刻停播。

    返回 True 表示播放期间被抢话（调用方应 `speaker.stop()` 并把用户那半句当下一轮输入）。
    """
    if listener.barge_in:
        return True
    end = time.time() + (cap_seconds if cap_seconds else max(10.0, len(answer or "") / 8))
    while time.time() < end:
        if listener.barge_in:
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
                               on_event=None, barge_ms: int = 300, timeout_s: float = 60.0) -> dict:
    """Phase 2/3 主循环：流式转写 → 流式应答（句级合成播放）→ 播放期间可抢话。

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
        listener = MicListener(source, EnergyVAD(), barge_ms=barge_ms)
    listener.start()
    splitter = splitter or SentenceSplitter()
    speaker = StreamingSpeaker(tts, sink or NullSink())

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
        hit = listener.barge_in or await_playback(speaker, listener, answer)
        if hit:
            stats["barge_ins"] += 1
            speaker.stop()
            emit("barge_in", partial_answer=answer)
        else:
            emit("answer_done", answer=answer)
        stats["answers"].append(answer)
        return answer, hit

    try:
        emit("ready", source=getattr(listener.source, "kind", "?"))
        last_activity = time.time()
        while True:
            if max_rounds and stats["rounds"] >= max_rounds:
                break
            if timeout_s and (time.time() - last_activity) > timeout_s:
                emit("timeout")
                break
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
                audio = listener.take_utterance()
                if (ev.elapsed or 0) < 200 or audio is None or len(audio) < int(0.1 * listener.source.samplerate):
                    emit("too_short", speech_ms=ev.elapsed)
                    continue
                text = stt.transcribe(audio).strip()
                if not text:
                    emit("empty")
                    continue
                stats["transcripts"].append(text)
                emit("user", text=text)
                if is_exit(text):
                    emit("exit")
                    break

                last_activity = time.time()
                _answer, hit = await _turn(text)

                # 抢话后：用户那半句已经被 listener 收着 → 直接进下一轮（不用再说一遍）
                if hit:
                    pending = listener.take_utterance()
                    seconds = len(pending) / listener.source.samplerate if pending is not None else 0.0
                    emit("barge_pending", seconds=round(seconds, 2))
                    if pending is not None and len(pending) > 0:
                        text2 = stt.transcribe(pending).strip()
                        if text2:
                            stats["transcripts"].append(text2)
                            emit("user", text=text2)
                            await _turn(text2)
                speaker.close()
            if not keep_alive and stats["rounds"] >= 1:
                break
    finally:
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
              stt_model: str = "base", stream: bool = True, file_loop: bool = False) -> dict:
    """命令行入口：`forge --voice [--audio-source mic|file:PATH] [--voice-rounds N] [--voice-sink null]`。

    `--audio-source file:xxx.wav` = **不用麦克风也能跑完整语音链路**（L2 自测/回归用）。
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
    sink_obj = NullSink() if sink == "null" else FFplaySink()
    print(f"🔊 语音模式（Phase 2/3 流式 + 打断）· 音频源 {audio_source} · 播放 {sink} · 抢话阈值 {barge_ms}ms")
    print("   （说「退出」/「exit」结束；说话时可直接插话打断）")
    return asyncio.run(streaming_voice_loop(agent, stt, tts, source=src, sink=sink_obj,
                                            max_rounds=rounds, barge_ms=barge_ms,
                                            on_event=lambda ev: print(f"   · {ev.kind} {ev.data or ''}",
                                                                      flush=True)))
