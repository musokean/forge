# 客户端执行器（#17：中心大脑 + 分布式手脚）

让服务器上的 Agent 操控客户端 PC。本质是 **Computer Use**——**中心大脑 + 分布式手脚**，
核心不是自己造远程桌面，而是复用现成组件（RustDesk 做远控底座、GUI 自动化做手、视觉模型做眼），
再用 **规划 / 执行 / 评估 / 监督 四角色分离**防幻觉（对齐 A25）。

```
        中心（服务器 / 你的机器）                       客户端（被控 PC，可多台）
┌──────────────────────────────────────┐        ┌────────────────────────────────┐
│ forge 工具层  executor_run / cua_task │        │  python executor_agent.py      │
│ 控制平面      forge/executor_hub.py   │◀──出站──│  forge/executor.py             │
│   · 资产目录（注册表 + 能力声明）      │  长轮询  │   · shell（走本机沙箱）         │
│   · 策略（白名单 + 分阶段放权 + 超时）  │  反向连接│   · read_file/write_file/list_dir│
│   · 命令状态机（seq 关联 + 审计）      │─────────▶   · screenshot（可选 GUI 依赖）  │
│ Computer Use  forge/cua.py            │  回结果  │   · input 鼠标键盘（可选）      │
│   规划者→执行者→评估者→监督者          │        │   · 客户端策略：jail/体积/白名单 │
└──────────────────────────────────────┘        └────────────────────────────────┘
```

## 为什么是「反向连接」

被控 PC **只出站**、不开放任何入站端口：客户端用 HTTP 长轮询（`/api/executor/poll`，默认挂起
25s）向中心要命令，拿到就本地执行并把结果 `POST /api/executor/result` 回去。
好处：不用碰目标机器的防火墙/NAT，客户端只要能访问中心地址即可；坏处：命令延迟取决于轮询
（空转时最多等一个 `poll_wait`，有命令时立刻返回）。

## 四角色分离（防幻觉的关键）

**别让一个模型又当大脑又当手还当裁判**。四个角色是**四次独立模型调用**，各有各的 system prompt，
可以在 `cua.roles` 里各绑不同模型，而且**信息不对称**：

| 角色 | 看得到什么 | 产出 |
|---|---|---|
| Planner | 任务 + 设备能力/可写目录 | 1-6 个可验证步骤 |
| Executor | 当前步骤 + 上一轮证据 | **一个**动作（cap/action/args/expect） |
| Evaluator | **只看到原始证据**（output/error/extra），看不到 Executor 的自述理由 | 该步（或整体任务）是否真的达成 |
| Supervisor | 失败轨迹 | 重新规划，或判定不可完成 |

配套闸门：步数上限 · 连续失败阈值 · 重规划上限 · **计划用尽时的「整体完成检查」**
（执行者经常忘记说 done，没有这道闸门会把已完成的任务判成失败——真机实测踩到过）。

## 安全（两层策略 + 审计，对齐 A06/A25）

| 层 | 管什么 |
|---|---|
| 中心侧 `executor.hub` | 能力白名单 · 分阶段放权 `readonly/low_risk/approval/closed_loop` · 命令超时 · 结果体积上限 · 每设备待发队列上限 · **每条命令一条审计** |
| 客户端侧 `executor.client` | 能力白名单（**未装 GUI 依赖就不声明 screenshot/input**）· 路径 jail（`root`，防 `../` 与符号链接逃逸）· 读写体积上限 · shell **走本机沙箱**（`forge/sandbox.py`：Docker 可用则容器隔离，否则加固本机执行） |

分阶段放权（A25「先做内部团队，再放外部」）：`readonly`（只读）→ `low_risk`（低风险写，默认）→
`approval`（写类动作要 `approved=true` 人工确认）→ `closed_loop`（都成熟后再全放）。
写类工具本身还走 forge 的审批层——**策略拒绝发生在命令出中心之前**。

## 跑起来

```bash
# 中心（任意一台能跑 forge 的机器）
forge --serve                      # 默认 127.0.0.1:8080；对外服务请配 server.api_keys

# 被控 PC（每台一个；只出站）
pip install "handcraft-agent[executor]"     # 可选：装了才有 screenshot / input 能力
python executor_agent.py --center http://<中心地址>:8080 --token <APIKEY> --id pc-01 \
       --root D:/允许操作的目录
python executor_agent.py --list-caps        # 看本机能声明哪些能力（没装 GUI 依赖会如实显示）

# 中心侧：REPL 里
/executor                        # 枢纽概览 + 在线执行器
/executor list
/executor run pc-01 "whoami"      # 在目标 PC 上跑命令
/executor cua pc-01 "打开记事本并输入 hello"   # 四角色闭环（需要 GUI 能力）
/executor audit 10                # 命令审计
```

Agent 也可以直接用工具：`executor_list` / `executor_run` / `executor_file` /
`executor_screen` / `executor_input` / `cua_task`。

## HTTP 接口（都是同一套鉴权）

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/executor/register` | 客户端注册（身份 + 能力声明） |
| POST | `/api/executor/poll` | 客户端长轮询领命令（兼心跳） |
| POST | `/api/executor/result` | 客户端回结果 |
| GET | `/api/executor/devices` | 在线执行器 + 枢纽统计 |
| POST | `/api/executor/command` | 派一条命令并等结果 |
| POST | `/api/executor/cua` | 四角色 Computer Use 任务 |

## 与 RustDesk 的关系（A25：别自己造远程桌面）

本模块**不实现远程桌面协议**。分工：

- **RustDesk**：人在应急时接管画面（自带 ID/密码、P2P、端到端加密）。
- **本模块**：Agent 在结构化能力（命令/文件/截屏/输入）上闭环，做「看→决策→操作→验证」。
- 需要「Agent 看着 RustDesk 窗口操作」时，用 `input` + `screenshot`（中心侧拿截图，接视觉模型判断）。

## 当前状态（诚实标注）

| 项 | 状态 |
|---|---|
| 枢纽 / 客户端 / 控制平面 / HTTP 接口 / 工具 / CLI | ✅ 已落地，`test_executor.py` 38 例全离线通过 |
| 端到端（中心服务 + 真客户端长轮询 + 真派命令 + 远端文件读写 + jail 拦截） | ✅ 本机实测（含独立核对磁盘产物） |
| 四角色闭环（真模型） | ✅ 本机实测：3 步 / 13.2s 完成「远端建文件并读回确认」；**首轮实测暴露的三个缺口已修**（完成闸门、可写目录进简报、执行者提示词） |
| GUI 能力（screenshot / input） | ⬜ 未在真机验证：本机没装 pyautogui/Pillow，客户端**如实不声明**这两个能力。装上 `[executor]` extra 即可用，但「看屏幕理解」还需要中心侧接视觉模型 |
| 多台被控 PC 统一调度 | ✅ 结构支持（注册表按 `device_id` 隔离、每设备独立队列）；未做实机多机压测 |
| 外部客户（不可信）场景 | ⬜ 需要安全拉满：强鉴权 + 每步审批 + 会话加密 + 白名单设备（A25「分级安全」） |
