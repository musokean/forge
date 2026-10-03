# 硬件链路（#16 硬件 Phase 1）

把美容仪这类外设接进 forge：**硬件即工具**，Agent 通过协议控设备。Phase 0（`fake_device.py`）
是进程内模拟器；Phase 1 是**真链路**——串口 / MQTT + 一层控制平面。

```
Agent 工具层      tools.py: device_status / device_power / device_level / device_reset
    ↓
控制平面          forge/hwcontrol.py   资产目录 · 策略引擎 · 命令状态机（Created→Sent→Accepted→Applied）
    ↓
协议              forge/hwproto.py     line-JSON + CRC + seq/ack/state（协议 v1）
    ↓
传输              forge/hwtransport.py 串口(pyserial) / MQTT(paho) / 内存
    ↓
设备              device_sim.py（模拟） · hardware/esp32_beauty_device.ino（真机骨架）
```

## 协议 v1

一条消息 = 一行 JSON（UTF-8，`\n` 结尾）。串口与 MQTT **共用同一套帧**。

```json
{"v":1,"seq":7,"ts":1759000000.12,"type":"cmd","id":"set_level","args":{"level":2},"crc":"3f9a1c2b"}
```

| 字段 | 含义 |
|---|---|
| `v` | 协议版本（当前 `1`；不符直接拒绝 `E_VERSION`） |
| `seq` | 序号，设备回复原样带回 → 用来把回复配到命令上 |
| `ts` | 时间戳（float 秒，可选） |
| `type` | `hello` / `hello_ack` / `cmd` / `ack` / `state` / `event` / `ping` / `pong` |
| `crc` | `crc32(不含 crc 字段的规范化 JSON)` 取前 8 位 hex |

**规范化** = 键按字典序、分隔符无空格（等价 `json.dumps(body, sort_keys=True, separators=(",",":"))`）。
串口线上会掉字节，没有校验就会出现「发的是 2 档，设备跳成 3 档」这类**静默错控**——
这也是固件里必须按同样顺序输出字段的原因。

### 命令时序（ACK 与 Applied 是两件事）

```text
Agent                                    设备
  │  hello  (协议版本 + 客户端)            │
  │ ─────────────────────────────────────►│
  │  hello_ack (device_id/name/fw/能力/限额/安全)  │
  │ ◄─────────────────────────────────────│
  │  cmd {id:"set_level", args:{level:2}} │
  │ ─────────────────────────────────────►│
  │  ack  {ok:true}          ← 设备「接下了」
  │ ◄─────────────────────────────────────│
  │  state {power:true, level:2, ...}     ← 设备「真实状态」
  │ ◄─────────────────────────────────────│
```

- `ack` 只说明设备接受了请求；**只有 `state` 快照才算 applied**。
- 设备也可以只回 `state`（简单固件），控制平面同样认。
- 超时 → 重试（`device.retries`）→ 仍失败则 `Timeout`，并按需**回滚**（例如档位回到上一档）。
- 过热等异步事件用 `event` 上报（`kind: "over_temp"`），控制平面会记审计 + 触发断电。

### 错误码

| 码 | 场景 |
|---|---|
| `E_BAD_JSON` | 不是 JSON / 缺 `crc` / 非 UTF-8 |
| `E_CRC` | 校验不符（线路噪声、半帧错位） |
| `E_VERSION` | 协议版本不符 |
| `E_UNKNOWN_TYPE` | 未知消息类型 |
| `E_UNSUPPORTED` | 设备不支持该命令 |
| `E_REJECTED` | 设备明确拒绝（未开机/越界/需复位…） |
| `E_TIMEOUT` | 命令超时（含重试用尽） |
| `E_NO_STATE` | 收到 ack 但没等到 state（未确认生效） |

## 承载无关

同一套协议跑在三种承载上，换承载不改上层代码：

| 承载 | 配置 | 用在哪 |
|---|---|---|
| 串口 / UART | `transport: serial` + `serial_url: COM5` | 台式设备、USB-TTL 调试口 |
| TCP 透传 | `serial_url: socket://127.0.0.1:9009` | 串口服务器、esp-link、**没有硬件时的自测** |
| MQTT | `transport: mqtt` + `device.mqtt.host/port` | 局域网多设备（WiFi/IoT） |
| 回环 | `serial_url: loop://` | 只验传输层 |

MQTT 主题约定：

```text
下行  {prefix}/cmd/{device_id}     # hello / cmd / ping
上行  {prefix}/up/{device_id}      # hello_ack / ack / state / event / pong
```

## 没有硬件也能验真链路

```bash
# 终端 1：模拟设备端（说真协议，TCP 透传服务端）
python device_sim.py --transport socket --port 9009 --test-hooks

# 终端 2：Agent 侧连上去（走真 pyserial + 真协议 + 真控制平面，只差物理器件）
python main.py
/device mode serial socket://127.0.0.1:9009
/device connect
/device                    # 状态 + 策略 + 审计
/device audit 5
```

也可以直接把 `device.enabled: true` + `serial_url: socket://127.0.0.1:9009` 写进配置，然后正常对话——
Agent 调 `device_level` 就走真链路。`device_sim.py --test-hooks` 提供 `sim_set_temp` / `sim_trip`
用来快速制造过热场景（**仅自动化验证用，别在真机上开**）。

接真设备：`/device mode serial COM5`（Windows）或 `/dev/ttyUSB0`（Linux），波特率与固件一致（默认 115200）。

## 控制平面（生产级的真正难点）

直接让 Agent 发消息 = 演示，不是生产。缺了控制平面会出三件事：设备别名漂移 / 跨站点误控 /
ACK 语义不一。所以这一层管三件事：

| 层 | 实现 | 说明 |
|---|---|---|
| 资产目录 | `AssetRegistry` | 别名 → 设备身份/站点；**未知别名一律拒绝，绝不猜**（多设备时必须点名） |
| 策略引擎 | `PolicyEngine` | 分阶段放权 `readonly / low_risk / approval / closed_loop` + 限额（档位/温度/时长）+ 写冷却 + **远程（非本机）默认不可写** |
| 命令服务 | `CommandService` | 状态机 + 超时 + 重试 + 回滚 + 审计（每条命令一条结构化日志） |

**分阶段放权**（别一上来就闭环）：`readonly`（只读）→ `low_risk`（低风险写，默认）→
`approval`（中风险人工确认）→ `closed_loop`（权限/审计/ACK/超时/回滚都成熟后才自动执行）。

写命令还会**叠加 forge 原有的审批层**（`device_power` / `device_level` / `device_reset` 都是
`read_only=False`）——策略拒绝发生在命令**发出去之前**，审批发生在工具调用之前，两层都要有。

## 配置

```yaml
device:
  enabled: true               # false = Phase 0 模拟器（默认）
  transport: serial           # sim | serial | mqtt
  device_id: beauty-01
  serial_url: "socket://127.0.0.1:9009"   # 或 COM5 / /dev/ttyUSB0 / loop://
  baudrate: 115200
  timeout: 3
  retries: 2
  monitor: true               # 后台巡检：限额执行（超温/超时主动断电）+ 心跳
  monitor_interval: 5
  actor: agent                # 审计里的操作者
  assets: {}                  # 多设备：{别名: {device_id, transport, site}}
  policy:
    stage: low_risk
    max_level: 3
    max_temp_c: 45
    max_runtime_s: 900
    cooldown_s: 1
    allow_remote: false
```

## 真机接线（ESP32 参考）

见 `hardware/esp32_beauty_device.ino`（同一套协议，换成真继电器/PWM/温度传感器）。
接线示例：档位 → GPIO 25/26/27，电源 → GPIO 14，NTC 温度 → GPIO 34。

> ⚠️ **强电安全**：控市电设备必须做隔离（继电器/光耦 + 独立电源），调试用低压假负载先跑通链路再上真机。
> Agent 侧的控制平面 + 设备端的过热断电是两道独立防线，**两边都要保留**，别只依赖一边。

## 当前状态（诚实标注）

| 项 | 状态 |
|---|---|
| 协议 / 传输 / 控制平面 / 工具接线 / `/device` | ✅ 已落地，`test_hardware.py` 36 例全离线通过 |
| 真链路端到端（TCP 透传 + pyserial `socket://`） | ✅ 已在本机实测（含断线重连、越权拦截、过热守卫） |
| 真串口 COM 口 + 真美容仪 | ⬜ 需要硬件（本机无；换 `serial_url` 即可，代码路径同） |
| MQTT 对真 broker（mosquitto 等） | ⬜ 未验（单测用假 broker 客户端验证了主题/载荷/回调） |
| ESP32 固件 | ⬜ 参考骨架，未编译/未上机 |
