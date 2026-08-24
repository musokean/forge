"""语音交互模块（M3 #11，Phase 1：跑通链路）。

链路：录音 → STT 转文字 → Agent.run()（复用现有能力）→ TTS 合成 → 播放。

设计原则：
  · 语音只换「输入输出」，不侵入 Agent 核心——本模块独立，Agent 零改动。
  · 依赖延迟导入（sounddevice / whisper / edge_tts 都是函数内 import）——
    forge 核心（168 测试）不装语音依赖也能跑；缺依赖时 forge --voice 给出友好安装提示。
  · 可插拔：STT/TTS 各一个接口，默认实现可换（Whisper→FunASR、Edge-TTS→CosyVoice）。

Phase 1 范围（2026-08-24 方案）：
  · 整句级（说完整句再答），不做流式 STT / barge-in 打断（二期）。
  · 录音端简单能量静默检测（0.8s 无声 = 说完）。
"""
import asyncio
import importlib
import importlib.util
import os
import sys
import tempfile
import time

# 语音依赖安装提示（用户装一次）
_REQUIRED = {
    "edge_tts": "pip install edge-tts",
    "sounddevice": "pip install sounddevice",
}
_STT_HINT = "pip install openai-whisper   # 或换其他 STT（见 voice.py 顶部说明）"


def _check_deps() -> list[str]:
    """检查语音依赖，返回缺失项列表（空 = 全齐）。"""
    missing = []
    for mod, cmd in _REQUIRED.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(f"{mod}（安装：{cmd}）")
    if importlib.util.find_spec("whisper") is None:
        missing.append(f"whisper（安装：{_STT_HINT}）")
    return missing


# ===================== STT：语音 → 文字 =====================
class STTEngine:
    """语音识别抽象：子类实现 transcribe(audio_path) -> str。"""

    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class WhisperSTT(STTEngine):
    """Whisper 本地识别（Phase 1 默认，中文可用 base/small）。"""

    def __init__(self, model="base"):
        import whisper  # 延迟导入（重依赖）

        self._model = whisper.load_model(model)

    def transcribe(self, audio_path: str) -> str:
        result = self._model.transcribe(audio_path, language=None)
        return (result.get("text") or "").strip()


# ===================== TTS：文字 → 语音 =====================
class TTSEngine:
    """语音合成抽象：子类实现 synthesize(text, out_path) -> str（返回音频路径）。"""

    def synthesize(self, text: str, out_path: str) -> str:
        raise NotImplementedError


class EdgeTTS(TTSEngine):
    """Edge-TTS（免费，中文自然，无需 key）。"""

    def __init__(self, voice="zh-CN-XiaoxiaoNeural", rate="+0%", volume="+0%"):
        self._voice = voice
        self._rate = rate
        self._volume = volume

    def synthesize(self, text: str, out_path: str) -> str:
        import edge_tts  # 延迟导入

        async def _run():
            communicate = edge_tts.Communicate(text, self._voice, rate=self._rate, volume=self._volume)
            await communicate.save(out_path)

        asyncio.run(_run())
        return out_path


# ===================== 录音（sounddevice → wav） =====================
def record_until_silence(
    out_path: str,
    samplerate: int = 16000,
    silence_seconds: float = 0.8,
    max_seconds: float = 30.0,
) -> float:
    """录音直到静默（能量阈值判定）或超时，返回实际录音时长（秒）。

    sounddevice 需要真实麦克风——沙箱无麦克风，本机验证用。
    """
    import sounddevice as sd
    import numpy as np

    block = int(samplerate * 0.2)  # 200ms 一块
    threshold = 0.01  # RMS 能量阈值（调参点）
    chunks = []
    silent_blocks = 0
    max_blocks = int(max_seconds / 0.2)
    required_silent = max(1, int(silence_seconds / 0.2))

    # 等首个有声块（说之前不录音）
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
    import wave

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return len(audio) / samplerate


# ===================== 播放（ffmpeg → 系统音频） =====================
def play_audio(path: str) -> None:
    """播放音频（优先 ffplay，失败用系统方案）。沙箱无音频设备，本机验证用。"""
    import shutil
    import subprocess

    ffplay = shutil.which("ffplay")
    if ffplay:
        try:
            subprocess.run([ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=True)
            return
        except Exception:
            pass
    # 兜底：Windows 系统播放（winsound），Unix 用 aplay
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME)
    else:
        subprocess.run(["aplay", path], check=True)


# ===================== 语音对话主循环 =====================
async def voice_loop(agent, stt: STTEngine, tts: TTSEngine, keep_alive=True, max_rounds: int = 0) -> None:
    """整句级语音对话：说一句 → 答一句。

    agent      : forge 的 Agent 实例（复用全部能力：路由/工具/知识库/记忆）
    stt        : STT 引擎（WhisperSTT 等）
    tts        : TTS 引擎（EdgeTTS 等）
    keep_alive : False 时跑一轮就退出（测试用）
    max_rounds : 最多迭代轮数（0 = 无限；测试防死循环用）
    """
    print("🔊 语音模式已开启（说「退出」/「exit」结束）")
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
            text = stt.transcribe(wav)
            text = text.strip()
            if not text:
                print("（识别为空，再说一次）")
                continue
            print(f"🗣 你说：{text}")
            if text in ("退出", "exit", "结束", "再见"):
                print("👋 语音模式结束")
                break
            answer = await agent.run(text)
            print(f"🔨 forge：{answer}")
            out_mp3 = os.path.join(tmp, "reply.mp3")
            tts.synthesize(answer, out_mp3)
            play_audio(out_mp3)
        if not keep_alive:
            break


def run_voice(agent) -> None:
    """命令行入口：forge --voice。缺依赖时给友好提示，不崩。"""
    missing = _check_deps()
    if missing:
        print("❌ 语音模式需要以下依赖（装一次即可）：")
        for m in missing:
            print(f"   · {m}")
        print("   安装示例：pip install edge-tts sounddevice openai-whisper")
        return
    stt = WhisperSTT()
    tts = EdgeTTS()
    asyncio.run(voice_loop(agent, stt, tts))
