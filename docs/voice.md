# 语音交互（#11：Phase 1 整句级 → **Phase 2 流式 + Phase 3 打断**）

```
麦克风 ──▶ VAD 分段 ──▶ STT 转写 ──▶ Agent ──▶ TTS 合成 ──▶ 播放
   ▲                    (边说边出草稿)   │        (一句一合成)
   └────────── 播放/生成期间仍在听 ───────┴──▶ 检测到插话 → 立刻停播 + 取消生成
```

**Phase 1（既有）**：说一句 → 静音自动结束 → 整段转写 → 答完 → 整段合成 → 播放。
一句一答能跑，但**首句出声延迟 = 整段答案生成 + 整段合成**，且它说话时你插话没用。

**Phase 2 流式**：
- **转写流式**：说话期间每隔 `partial_ms`（默认 1s）把已采集的音频重转一次，先出草稿
  （真机实测：`你好,请用` 在整句说完前就出来了）。
- **应答流式**：Agent 的增量文本进 `SentenceSplitter` 切句，**合成一句就播一句**，
  首句出声不再等整段答案（真机实测：一次回答切出 3 句、按序播放）。

**Phase 3 打断（barge-in）**：forge 说话/生成期间麦克风**仍在听**：
1. 检测到持续说话 ≥ `barge_ms`（默认 300ms）→ **立刻停播**（`StreamingSpeaker.stop()` 打断当前播放并丢弃未播队列）；
2. **取消当前生成**（`Agent.run(interrupt_check=...)`，复用既有中断语义：返回已生成部分）；
3. **保住已经说的那半句**（`MicListener.take_utterance()`）→ 直接作为下一轮输入，**不用再说一遍**。

## 为什么能「不用麦克风、不出声、不调模型」就测

链路里三块都做成了可注入，机制与硬件解耦：

| 组件 | 真实现 | 测试替身 |
|---|---|---|
| `AudioSource` | `SoundDeviceSource`（真麦克风） | `FileSource`（音频文件当麦克风，遍与遍之间插静音供 VAD 切句）· `ScriptedSource`（合成「有声/静音」块，纯逻辑） |
| `AudioSink` | `FFplaySink`（ffplay 子进程，`stop()`＝terminate，打断的即时性来源） | `NullSink`（记录「本该播什么、什么时候被打断」） |
| VAD / 切句 | `EnergyVAD` · `SentenceSplitter` | 本身就是**纯状态机**，喂块即可断言边界 |

于是「分段边界 / 打断时序 / 流式顺序 / 首句延迟」全部能确定性复现，进 CI。
真机验收只剩最后一步（真人戴耳机说话），见下。

## 跑起来

```bash
pip install "handcraft-agent[voice]"          # sounddevice / numpy / edge-tts / openai-whisper

forge --voice                                  # Phase 2/3（默认）：真麦克风 + 真播放
forge --voice --voice-phase1                    # 退回 Phase 1 整句级
forge --voice --stt-model small                 # 中文识别要求高时换更大的 whisper 模型
forge --voice --barge-ms 500                    # 插话判定阈值（越小声的环境调大越稳）

# 不用麦克风也能跑完整链路（自测/回归、CI 友好）
forge --voice --audio-source file:问句.wav --voice-sink null --voice-rounds 1
forge --voice --audio-source file:问句.wav --voice-loop --voice-rounds 5   # 同一句连跑 5 轮
```

| 参数 | 作用 |
|---|---|
| `--audio-source mic\|file:PATH` | 音频来源；`file:` = 文件当麦克风（无需硬件） |
| `--voice-rounds N` | 最多几轮就退出（0=一直跑） |
| `--voice-loop` | 文件源循环播放（多轮回归用；默认不循环，避免把同一句当成插话） |
| `--voice-sink speaker\|null` | 播放方式；`null` = 只走流程不出声（自测用） |
| `--barge-ms N` | 连续说话多久算插话 |
| `--stt-model NAME` | whisper 模型（base/small/medium…） |
| `--voice-phase1` | 用 Phase 1 整句级链路 |

## 测试阶梯（哪一级需要什么）

| 级别 | 测什么 | 需要什么 | 结论 |
|---|---|---|---|
| **L1 离线机制** | VAD 分段 · 打断状态机 · 流式顺序与首句延迟 · 句级播放 | 注入式音频源 + 假播放器，无硬件无模型 | ✅ `test_voice.py` **39 例**进 CI（`test_interrupt.py` +2） |
| **L2 无麦闭环** | 真 STT → 真 Agent → 真 TTS 全链路、流式切句 | 音频文件当麦克风，静音 | ✅ 本机实测（真 whisper 草稿 + 真模型流式 + 3 次句级合成） |
| **L3 半真机** | 真采集路径、重采样、驱动缓冲、声学回声 | 扬声器放 + 麦克风收（会出声）；或装 VB-Cable 做数字回环 | ⚠️ 未做：本机「立体声混音」只在 WDM-KS 下暴露，PortAudio 打不开；声学回环会出声 |
| **L4 真机验收** | 打断体感、耳机回声、现场噪声下的阈值 | **人戴耳机说两句** | ✅ **2026-09-29 真机验收通过**（用户实测）：说一句答得对 · 它说话时插话立刻停并接上新话 · 首句出声明显更快 |

本机已验证：真麦克风可开（采到 1.4s，环境底噪 RMS≈0.005）· **15 秒环境噪声不误触发**（阈值默认 0.012）· 真 whisper `base` 中文可用但精度有限（实测「12 乘以 8」→「12×18」，要准确请 `--stt-model small` 或后续接 SenseVoice/FunASR）。

**首句出声延迟实测**（t=0 为转写完成，whisper base + edge-tts + 本机）：三句分别在 1.90s / 2.04s / 2.10s 切出 → **第一声 +4.18s**、第二声 +6.80s；整段生成在 **+7.23s** 结束——也就是说**老链路要到 7.23s 之后才开始出声**（还要再加首句合成时间，合计约 9.5s）。流式把首句出声从约 9.5s 提前到 **4.18s**。

## 已知限制（诚实标注）

- **外放 + 麦克风会自激**：TTS 被麦克风收回去 → 被当成新指令 → 死循环。**必须戴耳机**（这是 Phase 3 的硬性前提）。
- **whisper `base` 中文精度不够**：数字/近音词会错。`STTEngine` 是可插拔的，接大模型或 SenseVoice 是低成本改造。
- **转写流式是「重转写」而非真流式 ASR**：实现上每隔 `partial_ms` 对已采集音频重跑一次 whisper（CPU 上 base 约 1-2s/次）。好处是零新依赖、串口麦克风也能用；代价是草稿延迟约等于一次转写耗时。要更低延迟需换流式 ASR。
- **打断阈值是能量判定**：嘈杂环境下可能误触发（调大 `--barge-ms`），安静环境灵敏度足够。
- **L3 未做**：真声学回环（扬声器放 → 麦克风收，会出声）与数字回环（需装 VB-Cable，本机「立体声混音」在 PortAudio 下打不开）都还没跑。L4 真人验收已通过（见上表）。
- **麦克风设备选择**：目前用系统默认输入设备（`--audio-source mic`），还没有「指定输入设备」的参数——多麦克风的机器需要靠系统默认设备切换。
