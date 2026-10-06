# forge

**一个你读得完的 ReAct 智能体 —— 步进循环约 235 行直白 Python，另外 31 个模块是可跳过的可选层。**

**零重依赖**（标准库 + `openai` + `pyyaml`）。每个机制都摊开写，不藏着：重试 · 降级回退 · 熔断 · 写操作审批闸门 · 命令沙箱 · 结构化日志 · 四角色 Computer Use 循环。任何 OpenAI 兼容端点都能接 —— DeepSeek、通义千问、vLLM、Ollama、本地模型。

<!-- 录好 GIF 后取消这行注释、并删掉本注释：![forge REPL —— 提问、看执行轨迹、生成中随时打断](docs/assets/forge-cli.gif) -->


[![CI](https://github.com/musokean/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/musokean/forge/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/handcraft-agent.svg)](https://pypi.org/project/handcraft-agent/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.13-blue.svg)](https://github.com/musokean/forge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **English**: [README.md](README.md)

把 M0→M3 里程碑落成代码——ReAct 核心循环 + 工程底盘（工具只读分级 / 上下文截断+滚动摘要 / 错误重试+降级 / token 统计 / 每步 trace / 写操作审批）+ 多智能体编排（并行拆解 / 讨论式辩论 / 多模型路由 / 自动路由 / 流式输出）+ 结构化输出 + 本地知识库（SQLite+FTS5）+ 黄金集评估 + Web 界面 + HTTP API 服务（多会话 + 鉴权 + 限流）+ 工具安全沙箱（Docker 隔离/加固本机降级）+ 完整日志（结构化+脱敏）+ Markdown 导出。**模型配置驱动、零外部重依赖**。

---

## 一、快速开始

```bash
# 1. 安装（三选一）
#    ① 从 PyPI 装（推荐）
pip install "handcraft-agent[server,device]"
#    ② 直接从 GitHub 装（无需 clone）
pip install "git+https://github.com/musokean/forge.git"
#    ② 或 clone 后本地安装
#    git clone https://github.com/musokean/forge.git && cd forge && pip install -e .

# 2. 设置 API key（环境变量，自动读取）
export DEEPSEEK_API_KEY=sk-xxx

# 3. 启动交互式对话
forge

# 4. 单次问答（不进入交互模式）
forge "帮我算 (3+5)*2"

# 5. Web 网页界面（浏览器聊天，零依赖 HTTP 服务）
forge --web                  # 默认端口 8000，自动开浏览器
forge --web --port 8080      # 指定端口

# 6. HTTP API 服务（#14 部署：多会话 + 鉴权 + 限流）
#    先装可选依赖：pip install "handcraft-agent[server]"
forge --serve --port 8080    # 接口文档 http://127.0.0.1:8080/docs
```

> **首次运行**：找不到配置文件时自动生成默认 `config/models.yaml`（不崩、不报错），填 key 或设环境变量即可用。模型/角色/辩论阵容/知识库路径全在 `config/models.yaml` 里改，保存即生效，不动代码。
> **pip 安装的**：配置落在 `~/.forge/config/models.yaml`（也可用环境变量 `FORGE_CONFIG` 指定别的路径）；**当前目录下的 `config/models.yaml` 优先级最高**。

交互式对话里**直接说即可**，forge 自动判断任务类型：简单问题直接答、多任务自动并行拆解、决策类问题自动多角色辩论，不用手动指定。

> **forge 不是内部命令**？把 Python 的 Scripts 目录（通常 `C:\Users\你\AppData\Local\Programs\Python\Python3xx\Scripts`）加进系统 PATH。
>
> **想改名**：改 `main.py` 顶部 `NAME` + `pyproject.toml` 的 `[project.scripts]`，再重跑 `pip install -e .`。

---

## 二、命令手册（交互模式内使用）

| 命令 | 作用 |
|------|------|
| `/reset` | 清空对话上下文 |
| `/usage` | 查看 token 用量（prompt / completion / 累计） |
| `/trace` | 查看本次会话步骤流水（模型步 / 工具调用 / 耗时 / token） |
| `/kb` | 知识库管理（详见「五、知识库」） |
| `/export` | 导出当前对话为 Markdown（详见「六、导出」） |
| `/key` | 配 key 三用法：`/key sk-xxx` 直接贴 key 自动配给主力 · `/key 模型别名 sk-xxx` 指定模型 · `/key` 交互向导（选模型→贴 key→可选切主力） |
| `/model` | 一键切主模型：`/model 模型别名`（立即生效，保留对话） |
| `/config` | 配置中心（详见「四、配置」） |
| `/circuit` | 熔断状态查看 / 复位（详见「四、配置」⑥ 熔断段） |
| `/skill` | 技能包切换：`/skill` · `/skill on 名称` · `/skill off 名称` |
| `/memory` | 长期记忆管理：`/memory` · `forget 词` · `clear` · `stats` |
| `/remember 内容` | 显式记一条关于你的长期记忆 |
| `/task` | 自动任务（定时/周期执行）：`/task` 列表 · `add 名 调度 [--kb] 提示词` · `del 名` · `run 名` · `on/off 名` · `log [名]` · `clear` |
| `/eval` | 黄金集回归（防变笨）：`/eval` 全量 · `list` 列出 · `add 任务|关键词|分` 新增 · `<序号>` 单例 · `export` 导出报告 |
| `/web` | Web 网页界面：`/web` 拉起（浏览器访问）· `/web stop` 停止 |
| `/serve` | API 服务（#14 部署）：`/serve` · `/serve 端口` 起 HTTP API（多会话 + 鉴权 + 限流）· `/serve stop` 停止 |
| `/logs` | 日志（#7）：`/logs` 状态 · `tail [n]` 最近事件 · `errors [n]` 告警 · `path` 目录 · `clear` 清空 |
| `/sandbox` | 工具沙箱（#4）：`/sandbox` 状态 · `mode <auto\|docker\|local\|off>` 切换策略 · `test <命令>` 试跑看实际路径 |
| `/device` | 硬件设备（#16）：`/device` 状态 · `connect [url]` · `disconnect` · `mode <sim\|serial\|mqtt> [url]` 切承载 · `reset` 复位过热保护 · `assets` 资产目录 · `audit [n]` 命令审计 |
| `/executor` | 客户端执行器（#17）：`/executor` 概览 · `list` 在线执行器 · `run <设备> <命令>` 远端执行 · `cua <设备> <任务>` 四角色闭环 · `audit [n]` 命令审计 |
| `/help` | 查看帮助 |
| `/exit` | 退出（或直接输 `exit` / `quit`） |

**配 key / 换模型的最短路径**（不用进面板）：
```
/key sk-xxx                    # 直接贴 key → 自动配给当前主力模型，最常用
/key deepseek_v4_pro sk-xxx    # 指定模型配 key（别名可只输前缀，自动补全）
/key                           # 交互向导：选模型 → 贴 key → 可选一键切主力
/model deepseek_v4_pro         # 切为主模型，立即生效
```
启动时 forge 会自动检测常用角色的模型是否缺 key 并提示。

**生成中打断 / 引导**（2026-08-20： 需求）：
- 按 **Esc**：立即中断当前生成，已生成内容保留
- 按**任意键**：进入引导模式，输入一句话（如「简洁点」「换个角度」），forge 按引导重新生成
- 连续打断 3 次自动中止（防死循环）；辩论/并行子任务不响应打断（无人值守场景）

每条回复结束后自动打印**状态栏**，实时显示：`[角色] ⏱ 首字/总耗时 ｜ 📊 token 增量/累计 ｜ 🧠 上下文百分比`。

---

## 三、工具清单（13 个）

| 工具 | 用途 | 类型 |
|------|------|------|
| read_file | 读本地文件（超 10 万字符自动截断） | 只读 |
| write_file | 写本地文件 | 写操作* |
| edit_file | 局部替换文件内容 | 写操作* |
| list_files | 列目录 | 只读 |
| search_file | 文件内容检索 | 只读 |
| calculator | 安全算数学表达式（AST 白名单，禁 eval） | 只读 |
| run_command | 执行本地命令 | 写操作* |
| get_time | 当前日期时间 | 只读 |
| web_search | 联网搜索（Bing，免 key） | 只读 |
| web_fetch | 抓取网页正文 | 只读 |
| kb_search | 知识库全文检索 | 只读 |
| kb_add | 把知识直接写入知识库（对话即沉淀） | 写操作* |
| kb_ingest | 把文件/目录导入知识库索引 | 写操作* |

\* 写操作执行前会触发**审批层**（CLI 里红字询问 `[y/N]`，回车默认拒绝；并行/辩论等无人值守场景自动放行）。拒绝后 forge 会收到反馈并调整方案。

---

## 四、配置（全部 `config/models.yaml` 驱动，改完即生效）

模型 / 角色 / 辩论阵容 / 路由 / 知识库路径全部在 `models.yaml` 定义，**改配置不用改代码**。也可以直接在 forge 里敲 `/config` 用交互面板改（选中模型、引导输入 key、写回配置、热重载，无需重启）。

### ① models —— 模型注册表（本地自建 + 预置主流）

```yaml
models:
  qwen3_local:                  # 别名
    label: 千问3-8B·本地
    base_url: http://192.168.66.54:4000/v1
    api_key: sk-xxx             # 明文 key，或 api_key_env: 环境变量名
    model: qwen3-8b-awq         # 供应商真实模型 ID
    preset: true                # true=预置主流（面板分组用），本地自建可省略
```

已预置 9 个主流云端模型（DeepSeek V4 Pro/Flash、通义 qwen-plus/max、OpenAI GPT-4o/mini、Kimi K2、GLM-4 Plus、SiliconFlow DeepSeek-V3），均为 OpenAI 兼容接口。**选中缺 key 的模型时，面板会引导输入 key**（明文 `sk-xxx` 或 `env:环境变量名` 走环境变量），写回配置立即生效。

### ② roles —— 角色绑模型（dict 写法）

```yaml
roles:
  default: { model: qwen3_local, label: 主力 }
  fallback: { model: qwen3_local_b, label: 降级 }   # 主力失败时降级到谁
```

- 任何 OpenAI 兼容端点都能接（Kimi / GLM / 智谱 / 本地 vLLM / Ollama…）
- 加新角色：`roles` 加一行，代码里 `Agent(role="translate")` 即可用

### ③ router —— 自动路由判断角色

```yaml
router:
  role: chinese       # 用哪个角色判断任务类型（单答/并行/辩论）
```

### ④ debate —— 辩论阵容

```yaml
debate:
  rounds: 2           # 辩论轮数
  roles:
    - name: 正方       # 辩手（含「裁判」的自动当最终裁判）
      model: reasoning
      persona: 你代表正方立场…
```

### ⑤ knowledge —— 知识库路径

```yaml
knowledge:
  db_path: data/knowledge.db    # 相对项目根；也可写绝对路径
```

### ⑥ circuit_breaker —— 熔断（#5 熔断，故障隔离）

```yaml
circuit_breaker:
  failure_threshold: 3   # 某角色连续失败几次触发熔断
  cooldown: 30            # 熔断后冷却秒数，到期自动半开探测
  half_open_max: 1        # 半开状态放行几次探测
```
某角色（端点）连续失败达阈值 → 自动熔断（OPEN），后续调用瞬间跳过它去试下一个角色，不再傻等重试退避；冷却到期后自动进入半开探测，成功则恢复。运行期用 `/circuit` 查看状态、`/circuit reset <角色>` 手动复位。

### ⑦ memory —— 长期对话记忆路径

```yaml
memory:
  db_path: data/memory.db    # 跨会话用户画像记忆（与知识库定位不同，见「八」）
```

### ⑧ reflect —— 反思自纠错（A07 组合拳末环，默认关）

```yaml
reflect:
  enabled: false     # 默认关；开启后每次回复多花 1 次评审调用
  min_score: 6       # 评审 0-10，低于此触发带意见重答
  max_rounds: 1      # 最多修正几轮
  judge_role: fallback  # 评审角色（建议用便宜的降级模型）
```
开启后：答案生成 → 评审角色打分 → 低于阈值带着具体意见重答一轮 → 仍低分则接受现状（不无限烧钱）。失败静默保留原答案，绝不比不纠更差。

---

### ⑨ server —— 服务化（#14 部署）

```yaml
server:
  host: 127.0.0.1          # 对外服务改 0.0.0.0（改之前先配好 api_keys！）
  port: 8080
  api_keys: []             # 例：["sk-forge-abc", "env:FORGE_KEY_2"]；留空 = 仅本机可访问
  rate_limit_per_min: 60   # 每调用方每分钟次数（0 = 关闭限流）
  approve_mode: auto_reject  # 服务端无交互审批 → 默认拒绝写操作
  db_path: data/sessions.db
```
需要可选依赖 `pip install "handcraft-agent[server]"`（核心仍零重依赖）；启动 `forge --serve`，接口文档 `/docs`。

---

### ⑩ logging —— 完整日志（#7）

```yaml
logging:
  enabled: true        # 关掉则完全不落盘
  level: INFO          # DEBUG / INFO / WARNING / ERROR
  dir: data/logs       # 每天一个文件 forge-YYYYMMDD.jsonl
  keep_days: 14        # 超期自动清理
  max_mb: 20           # 单文件上限，超过切分 .1/.2
  console: false       # 同时打到 stderr（默认关）
```
一条事件一行 JSON（`ts/level/event/run_id` + 字段）。**`api_key` / `Authorization` / `sk-xxx` 写入前一律变 `***`**——部署后的日志不能泄漏密钥。

### ⑪ sandbox —— 工具安全沙箱（#4）

```yaml
sandbox:
  mode: auto             # auto：Docker 可用走容器、不可用降级加固本机
                         # docker：强制容器（无 Docker 直接拒绝执行，生产用这个）
                         # local：加固本机 / off：直通（仅调试）
  image: python:3.11-slim
  timeout: 30
  network: false         # 容器默认断网
  memory: 256m           # 资源上限（防 fork 炸弹/吃光内存）
  cpus: "1.0"
  pids_limit: 128
  mount_rw: false        # 工作目录默认只读挂载
  max_output: 100000
  deny_patterns: []      # 追加的危险命令正则（内置已有 rm -rf /、mkfs、shutdown…）
```

---

### ⑪ device —— 硬件链路（#16 硬件 Phase 1）

```yaml
device:
  enabled: false             # false = Phase 0 进程内模拟器；true = 走真链路
  transport: sim             # sim | serial | mqtt
  device_id: beauty-01
  serial_url: "socket://127.0.0.1:9009"   # 或 COM5 / /dev/ttyUSB0 / loop://
  baudrate: 115200
  timeout: 3                 # 单条命令等待时长
  retries: 2                 # 超时重试次数
  monitor: true              # 后台巡检：超温/超时主动断电
  monitor_interval: 5
  policy:
    stage: low_risk          # readonly | low_risk | approval | closed_loop
    max_level: 3             # 档位上限（越权命令在发出去之前就被拒）
    max_temp_c: 45           # 温度上限
    max_runtime_s: 900       # 单次运行时长上限
    cooldown_s: 1            # 写冷却（防连续猛调）
    allow_remote: false      # 非本机链路默认不可写
```

---

## 五、知识库（索引库即源文档）
forge 的知识库是**自持的**：知识直接沉淀进库内条目（SQLite + FTS5 全文检索），**不依赖外部源文件**。新用户零配置开箱即用——对话里说「记住这个 / 记到知识库」，forge 自动调 `kb_add` 写入；外部文件导入（ingest/sync）只是可选的补充通道。

**管理命令 `/kb`**：

```
/kb                       查看状态（库路径 / 文档数 / 库内条目数 / 字符数）
/kb add 标题|内容          直接写入知识条目（同标题覆盖更新）；不带参数则交互输入
/kb list                  列出库内条目（标题 / 字数 / 更新时间）
/kb search <词>            全文检索（中文子串匹配，带高亮摘要）
/kb delete <标题或编号>     删除库内条目
/kb export [标题]          库内条目导出为 Markdown（exports/kb/，可指定单条）
/kb ingest <路径>          导入外部文件/目录（可选通道）
/kb sync <路径>            同步目录并登记：新增/变更入库、源删除的孤儿索引清理
/kb sync                  重放所有已登记的同步目录
/kb path <新库路径>         切换索引库位置（写回配置，立即生效）
```

- **对话即沉淀**：forge 的工具里有 `kb_add`（写操作，走审批层），你说「记住 X」它就把 X 写入知识库，之后随时 `kb_search` 检索到
- **自动整理**：`/kb sync <路径>` 登记过的目录，每次启动 forge 自动静默同步（同步只动文件索引，绝不清库内条目）
- 中文检索用 FTS5 trigram 分词，2 字词自动 LIKE 兜底，保证召回
- 库路径配置见「四、⑤ knowledge」

---

## 六、导出 Markdown

```
/export 文件名.md        # 导出当前对话到 exports/ 目录
/export                 # 自动命名 对话-时间戳.md
```

导出内容结构化：用户/助手消息分节、工具调用标注工具名、工具结果用引用块、滚动压缩的早期对话标注「📎 早期对话摘要」，可直接在 Obsidian 中阅读。

---

## 七、目录结构

```
handcraft-agent/
├── config/models.yaml    # 全部配置（模型/角色/辩论/路由/知识库/熔断/记忆/反思）
├── config/golden.yaml    # 黄金集（#13 评估用例，/eval add 可扩展，首次运行自动生成）
├── forge/
│   ├── config.py         # 配置加载 + resolve_model（角色/别名解析）
│   ├── config_writer.py  # 安全写回配置（/config 面板底层）
│   ├── llm.py            # openai SDK 网关 + 重试 + 降级 + 流式 + 熔断
│   ├── tools.py          # 20 工具 + @tool 注册 + 只读分级 + 知识库工具
│   ├── hwproto.py        # #16 硬件协议 v1（line-JSON + CRC + seq/ack/state）
│   ├── hwtransport.py    # #16 承载：串口(pyserial) / MQTT(paho) / 内存
│   ├── hwcontrol.py      # #16 控制平面：资产目录 + 策略引擎 + 命令状态机
│   ├── executor_hub.py   # #17 执行器枢纽（注册表 + 策略 + 命令队列）
│   ├── executor.py       # #17 客户端执行器（能力 + 路径 jail + 长轮询）
│   ├── cua.py            # #17 四角色 Computer Use 循环（规划/执行/评估/监督）
│   └── executor_agent.py # #17 被控 PC 上的入口
│   ├── agent.py          # ReAct 循环 + 上下文管理 + 状态栏 + trace + 审批 + 反思
│   ├── orchestrator.py   # 并行 / 辩论 / 多模型路由 / supervisor 规划执行
│   ├── router.py         # 自动路由（单答/并行/规划/辩论 四类判断）
│   ├── structured.py     # 结构化输出（Pydantic 校验 + RoleBrief）
│   ├── trace.py          # 每步 trace（内存 + JSONL 落盘）
│   ├── approval.py       # 写操作审批层（交互/回调/自动）
│   ├── knowledge.py      # 知识库引擎（SQLite+FTS5）
│   ├── skills.py         # 技能包系统（提示词片段+工具白名单按需装配）
│   ├── memory.py         # 长期对话记忆（用户画像，跨会话召回）
│   ├── reflect.py        # 反思自纠错（评审打分 + 低分重答）
│   ├── circuit.py        # 熔断器（三态机 + 注册表）
│   ├── tasks.py          # 自动任务调度器（进程内调度，SQLite 持久化）
│   ├── eval.py           # #13 黄金集评估（关键词命中 + LLM-as-judge + 报告导出）
│   ├── web.py            # #12 Web 界面（零依赖 http.server + 内嵌聊天页）
│   ├── server.py         # #14 API 服务（FastAPI：多会话 + 鉴权 + 限流 + 访问日志）
│   ├── sandbox.py        # #4 工具安全沙箱（Docker 隔离 / 加固本机降级）
│   ├── logging_setup.py  # #7 完整日志（结构化 JSONL + 轮转 + 保留期 + 脱敏）
│   ├── keypress.py       # 生成期键盘轮询（Esc 中断 / 引导输入，跨平台）
│   ├── spinner.py        # 等待动画（旋转指示器，首字到达即停）
│   └── console.py        # 终端样式（ANSI + 中文对齐，零依赖）
├── main.py               # CLI 入口（欢迎页 + REPL + 全部命令 + --web 启动）
├── data/knowledge.db     # 知识库默认索引文件（自动创建）
├── data/memory.db        # 长期记忆默认库（自动创建）
├── data/tasks.db         # 自动任务库（任务定义 + 执行记录，自动创建）
├── exports/              # /export 导出目录（自动创建）
├── test_m0~m2.py         # 里程碑验收测试
├── test_stress*.py       # 四轮压测 + 多任务混跑变体
├── test_structured.py    # 结构化输出测试
├── test_context_mgmt.py  # 滚动摘要/工具裁剪测试
├── test_trace.py         # trace 测试
├── test_approval.py      # 审批层测试
├── test_knowledge.py     # 知识库测试
├── test_error_resilience.py  # 模型失败兜底测试（402 不崩 REPL）
├── test_circuit_breaker.py   # 熔断三态机 + 降级跳过测试
├── test_skills_memory_reflect.py  # 技能包/长期记忆/反思/主管 四模块测试
├── test_eval.py          # #13 评估测试（黄金集/判定/报告，全 mock）
├── test_web.py           # #12 Web 测试（起真实 HTTP 服务 + mock Agent）
├── pyproject.toml        # 安装配置（注册 forge 命令）
└── requirements.txt
```

## 八、技能包 · 长期记忆 · 反思 · 主管（M3 补齐四件套）

四个新能力，各管一层：

| 能力 | 模块 | 一句话价值 | 入口 |
|------|------|-----------|------|
| **技能包 Skill** | `forge/skills.py` | 预置「提示词片段 + 工具白名单」按需装配：`coding` 编程 / `writing` 写作 / `research` 调研 / `knowledge` 知识库，激活后只给模型相关工具（省 token + 减少误调） | `/skill` |
| **长期记忆** | `forge/memory.py` | 跨会话记住你是谁：说「我是/我喜欢/我习惯…」自动沉淀；每次提问自动召回相关记忆注入上下文——forge 不再是每次见面的陌生人 | `/memory` · `/remember` |
| **反思自纠错** | `forge/reflect.py` | A07 组合拳末环：答案生成后评审打分，低分带意见重答（默认关，`/config` 可开） | 配置 `reflect` |
| **Supervisor 主管** | `forge/orchestrator.py` | 路由从「只分类」升级「分派+合并」：复杂任务 planner 拆解 → 并行执行 → merger 合并最终答案；拆解失败自动降级直答 | 自动（路由判定 `plan`） |

自动路由现在分四类：`single` 直答 · `parallel` 并行拆解 · `plan` 规划执行（supervisor）· `debate` 多角色辩论。

**等待动画**：等 AI 回复不再干瞪眼——流式首字到达前、路由判断、supervisor 拆解/合并、非流式生成全程有旋转指示器（`⠋⠙⠹…`），首字到达即停；并行/辩论子任务自动隐藏（防刷屏）。

## 九、自动任务（#18 定时 / 周期执行）

让 forge 在运行期间**后台自动执行**周期/定时任务：进程内调度器（后台线程 + 独立事件循环跑 `Agent.run`），SQLite 持久化（`data/tasks.db`），关掉 `forge` 重启后保留并补跑离线期间到期的任务。

```bash
/task                                          # 列出全部自动任务
/task add 每日简报 每天09:00 帮我总结今天的重要事项
/task add 巡检 每2小时 检查知识库是否有过期条目       # 触发时自动执行该提示词
/task add 沉淀周报 每1天 --kb 生成本周要点并沉淀进知识库   # --kb：结果同时写进知识库
/task run 每日简报        # 立即手动跑一次
/task on 巡检 / off 巡检  # 启用 / 停用
/task log                 # 查看执行记录
/task del 每日简报        # 删除
```

调度类型：`每N小时` / `每N分钟` / `每N天` / `每天HH:MM` / `once 2026-08-20T14:00`（一次性）。每次执行结果写 `runs` 表（`/task log` 查看）；写操作在自动任务中自动放行（不弹交互审批）。

---

## 十、黄金集评估（#13，防变笨）

每次对 forge 做结构性改动（prompt / 角色 / 模型 / 路由 / 技能）后，跑一遍黄金集回归，验证核心能力没有退化（对应 A10）：

```bash
/eval                 # 跑全量黄金集（并发跑批，关键词命中 + LLM-as-judge 双通道判定）
/eval list            # 列出黄金集用例
/eval add 计算 12×12|144|6    # 新增用例：任务|关键词1,关键词2|最低分（写回 config/golden.yaml）
/eval 3               # 只跑第 3 个用例
/eval export          # 把最近一次回归结果导出为 Markdown（exports/eval/）
```

**判定双通道**（互补，任一不过即判失败）：
- ① **关键词命中**（程序化硬指标）：每个用例声明期望答案必须包含的关键词，全命中才算过；不依赖 LLM，断网也能跑。
- ② **LLM-as-judge**（软指标）：评审角色给答案打 0-10 分，低于用例 `min_score` 判失败；评审不可用时（返回 None）不扣分，只认关键词通道。

黄金集存 `config/golden.yaml`（首次运行自动生成内置 6 例，`/eval add` 或直接编辑文件扩展）。

---

## 十一、Web 网页界面（#12，浏览器聊天）

零依赖的本地聊天页面（标准库 `http.server`，不引 FastAPI——保持项目零重依赖哲学）：

```bash
forge --web                     # 启动（默认 127.0.0.1:8000，自动开浏览器）
forge --web --port 8080         # 指定端口
# 或交互模式里：/web 拉起 · /web stop 停止
```

- **接口**：`GET /` 聊天页面 · `POST /api/chat`（`{"message": "..."}` → `{"reply": "..."}`）· `POST /api/reset` 重置 · `GET /api/status` 模型状态
- **能力**：与 CLI 同一套 Agent（ReAct + 工具 + 记忆召回），长期记忆钩子自动生效
- **界面（2026-08-20 美化）**：浅蓝渐变主题 + 消息头像 + 打字机流式效果 + 简易 Markdown 渲染（粗体/行内代码/代码块/列表）+ 复制按钮 + 日期分隔线
- **快捷键**：`Enter` 发送 · `Shift+Enter` 换行 · `Ctrl+Enter` 发送 · `Esc` 停止生成（或清空输入）· `↑/↓` 翻历史输入
- **停止按钮**：生成中显示「■ 停止」，点击即中断本次生成
- **安全默认**：Web 端没有交互审批通道，**写操作（写文件/改文件/执行命令/写知识库）自动拒绝**并提示回 CLI 执行；只读能力（检索/计算/读文件/联网）完全可用
- 单会话（一个 Agent 实例，`/api/reset` 清空）；多会话留到 M5 部署（#14）再做
- 页面为**单文件内嵌**（CSS/JS 全内联），断网也能打开，浅蓝主题与 CLI 一致

---

## 十二、API 服务（#14 部署：FastAPI + 多会话 + 鉴权）

把 forge 变成可被程序调用的 HTTP 服务（配合 `/api/chat` 即一个「自带知识库 + 记忆 + 多模型路由」的对话后端）：

```bash
pip install "handcraft-agent[server]"     # 可选依赖：fastapi + uvicorn
forge --serve --port 8080                 # 或交互模式里：/serve 8080 · /serve stop
```

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/healthz` | 探活（**无鉴权**，供容器/负载均衡健康检查） |
| GET | `/api/status` | 模型 / 会话数 / 鉴权模式 / 限流配置 |
| POST | `/api/chat` | `{"message": "...", "session_id": "可选"}` → 回复 + token 用量 |
| POST | `/api/sessions` | 新建会话（`{"title": "可选"}`） |
| GET | `/api/sessions` | 会话列表（含轮数、最后活跃时间） |
| GET | `/api/sessions/{id}` | 会话详情 + 完整消息历史 |
| DELETE | `/api/sessions/{id}` | 删除会话 |

**四条设计取舍（对齐 A12 部署与服务化）**：

- **会话持久化**：会话与消息存 `data/sessions.db`，每个会话独立 Agent 上下文——客户端断开后能接着聊，服务重启（同库）也恢复历史。`#12` 的 Web 是单会话，服务化必须多会话。
- **鉴权默认安全**：key 来自 `server.api_keys` 或环境变量 `FORGE_API_KEY`（支持 `env:变量名` 间接引用，key 不落配置文件）；**一个 key 都不配时自动退化为「仅本机 127.0.0.1 可访问」**——本地开发零配置，对外服务必须配 key，避免裸奔的 Agent。
- **限流**：按调用方（有 key 用 key 指纹，无 key 用来源 IP）滑动窗口 `rate_limit_per_min`，超限 `429` + `Retry-After`。
- **写操作默认拒绝**：服务端没有交互审批通道 → 写工具一律拒绝并把原因回报给调用方（与 #12 Web 一致）；只读能力（检索/计算/读文件/联网）全可用。受控部署可用 `server.approve_mode` 调整。

接口文档：`http://127.0.0.1:8080/docs`（FastAPI 自动生成的 Swagger UI）。调用示例：

```bash
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"message": "你好"}' http://127.0.0.1:8080/api/chat
```

---

## 十三、沙箱与日志（#4 / #7）

### 一、工具安全沙箱（#4）

`run_command` 不再直接在裸机上跑。策略看 `config/models.yaml` 的 `sandbox.mode`：

| 模式 | 行为 |
|------|------|
| `auto`（默认） | Docker 可用 → 容器隔离；不可用 → **加固的本机执行** |
| `docker` | 只用容器；本机没有 Docker 就**拒绝执行**（生产部署用这个） |
| `local` | 加固的本机执行（不需要 Docker） |
| `off` | 直通（等同旧行为，仅本地调试） |

容器约束：`--rm --network=none`（默认断网）、内存/CPU/PID 上限、根文件系统只读 + `/tmp` 可写、非 root（`65534`）、工作目录**只读挂载**、超时后自动 `docker rm -f` 清容器。降级到本机时仍有实际防线：**环境变量白名单**（命令读不到 `DEEPSEEK_API_KEY` 这类宿主密钥）、危险命令模式拦截（`rm -rf /`、`mkfs`、`dd` 写裸设备、`shutdown`…）、超时强杀、输出截断。

交互里可自查：`/sandbox`（状态）· `/sandbox mode docker`（切策略，立即生效）· `/sandbox test echo hi`（试跑一条，看实际走哪条路径）。

> 能力边界：沙箱管「执行的副作用与密钥泄漏」，**审批层（#4 审批）管「要不要执行」**——两层叠加才是完整的工具安全防线。

### 二、完整日志（#7）

每次 run、每个 HTTP 请求落一行 JSON 到 `data/logs/forge-YYYYMMDD.jsonl`：

```json
{"ts":"2026-09-28T23:31:34.123","level":"INFO","event":"run_end","run_id":"075ca629","role":"default","model":"deepseek-v4-flash","ms":3475.0,"prompt_tokens":812,"completion_tokens":96,"steps":3,"tools":["calculator({...})"]}
```

- **轮转与保留**：按天分文件，单文件超 `logging.max_mb` 切分，超 `logging.keep_days` 自动清理。
- **脱敏**：字段名含 `key`/`token`/`secret`/`authorization` 的值、以及 `sk-…` / `Bearer …` / `gho_…` 形态的字符串，一律写 `***`——**部署后的日志不能泄漏密钥**。
- **查看**：`/logs`（状态）· `/logs tail 20`（最近事件）· `/logs errors`（告警以上）· `/logs path` · `/logs clear`。

---

## 十四、客户端执行器（#17：中心大脑 + 分布式手脚）

让服务器上的 Agent 操控客户端 PC。**复用现成组件**（RustDesk 做远控底座、GUI 自动化做手），
核心是用**四角色分离**防幻觉（对齐 A25）。

```
中心（服务器）  executor_hub.py：注册表 + 策略 + 命令状态机 + 审计     ◀──出站长轮询──  被控 PC
                cua.py：规划者 → 执行者 → 评估者 → 监督者          ──派命令/回结果──▶  executor_agent.py
                                                                                      shell / 文件 / 截屏 / 输入
```

**反向连接**：被控 PC **只出站**（HTTP 长轮询，默认挂 25s），不开放任何入站端口，不用改防火墙。

**四角色防幻觉**（A25「别让一个模型又当大脑又当手还当裁判」）：四个角色是**四次独立模型调用**，
可在 `cua.roles` 各绑不同模型，而且**信息不对称**——评估者**只看原始证据**（命令输出/错误/文件内容），
看不到执行者的自述理由；再配步数上限 / 连续失败阈值 / 重规划上限 / **计划用尽时的整体完成检查**。

**两层策略 + 审计**：中心侧（能力白名单 + 分阶段放权 `readonly/low_risk/approval/closed_loop`
+ 超时 + 体积上限 + 每命令审计）· 客户端侧（**路径 jail** + 体积上限 + 能力白名单 + `shell` 走本机沙箱）。
**没装 GUI 依赖就不声明 `screenshot`/`input`**——不假装有这个能力。

```bash
forge --serve                                   # 中心
# 被控 PC（每台一个）
pip install "handcraft-agent[executor]"         # 可选：装了才有截屏/鼠标键盘
python executor_agent.py --center http://<中心>:8080 --token <KEY> --id pc-01 --root D:/work
# 中心 REPL
/executor · /executor list · /executor run pc-01 "whoami" · /executor cua pc-01 "打开记事本输入 hello"
```

Agent 侧工具：`executor_list` / `executor_run` / `executor_file` / `executor_screen` /
`executor_input` / `cua_task`。协议、安全与限制见 `docs/executor.md`。

---

## 十五、硬件链路（#16 硬件 Phase 1：串口 / MQTT 真链路）

硬件即工具。`tools.py` 暴露 `device_status` / `device_power` / `device_level` / `device_reset`；
背后接什么由配置决定：

| `device.enabled` / `transport` | 工具实际连到 |
|---|---|
| `false` 或 `sim`（默认） | Phase 0 进程内模拟器（`fake_device.py`） |
| `true` + `serial` | 真串口 / UART（`COM5`、`/dev/ttyUSB0`，或 `socket://host:port` 透传） |
| `true` + `mqtt` | MQTT：下行 `{prefix}/cmd/{device}`，上行 `{prefix}/up/{device}` |

**控制平面**（生产级的真正难点）：命令上线前要过三道——**资产目录**（未知别名一律拒绝，不猜）、
**策略引擎**（分阶段放权 `readonly`/`low_risk`/`approval`/`closed_loop` + 档位/温度/时长限额 +
写冷却 + 远程默认禁写）、**命令状态机**：

```
Created ──发送──> Sent ──ack.ok──> Accepted ──state──> Applied
                       └─ ack 拒绝 ─> Rejected
                       └─ 超时重试用尽 ─> Timeout（随后回滚）
```

`ack` 只代表设备**接下**了请求；**只有 `state` 快照才算生效**。每条命令都有审计，且 Agent 侧
自己也执行超温/超时断电守卫（不只依赖设备端保护）。

**没有硬件也能验真链路**（设备端模拟器说同一套协议，走 TCP 透传）：

```bash
# 终端 1：模拟设备端
python device_sim.py --transport socket --port 9009 --test-hooks
# 终端 2：forge 里
/device mode serial socket://127.0.0.1:9009
/device connect
/device                 # 设备身份 / 策略 / 最近审计
/device audit 10
```

这条路走的是真 pyserial + 真帧格式 + 真策略 + 真状态机，只差物理器件。协议规范与真机固件骨架见
`docs/hardware.md` 与 `hardware/esp32_beauty_device.ino`。

---

## 十六、语音交互（#11：Phase 2 流式 + Phase 3 打断）

```
麦克风 ─▶ VAD 分段 ─▶ STT（边说边出草稿）─▶ Agent（边生成边切句）─▶ TTS 逐句合成 ─▶ 播放
   ▲                                                                              │
   └──────────────── 播放/生成期间仍在听：插话即打断 ◀──────────────────────────────┘
```

```bash
pip install "handcraft-agent[voice]"     # sounddevice / numpy / edge-tts / openai-whisper
forge --voice                             # 说话，并在它回答时直接插话
```

- **Phase 2 流式**：说话期间每秒左右重转一次，**草稿在你没说完时就出来**；回答边生成边切句，
  **合成一句就播一句**——首句出声不再等整段答案（真机实测：一次回答切 3 句、按序播放）
- **Phase 3 打断**：它思考/说话时麦克风**仍在听**；你一开口 → **立刻停播 + 取消当前生成 +
  已说的半句直接进下一轮**（不用再说一遍）；插话发生在**播放期间**同样能打断
- **可测性是设计出来的**：麦克风与播放器**可注入**（真麦克风 / 音频文件当麦克风 / 脚本化合成音频；
  真 ffplay 或**只记录不发声**的假播放器），VAD 与切句是**纯状态机** →
  「分段边界 / 打断时序 / 流式顺序」都在 CI 里确定性跑，**不需要麦克风、模型、出声**
- **没麦克风也能试**：`forge --voice --audio-source file:问句.wav --voice-sink null --voice-rounds 1`
  能静音跑完真 whisper + 真模型 + 真 edge-tts 全链路
- **回声消除（`--aec`）= 真免手**：它说话时麦克风照样开着，用「正在播的音频」当参考把回声减掉 →
  **不闭麦、不按键也能随时插话**。默认纯 numpy 的 NLMS（无 C 扩展依赖）；模式内自动改用进程内播放
  （`ffplay` 拿不到正在播的样本）。
- **外放也能用（不必戴耳机）**：`forge --voice --half-duplex` = 它说话时闭麦（**思考期间仍可插话**，那时没有回声）；
  `forge --voice --ptt` = **按住空格说话**，按下即打断、松开提交。戴耳机全双工仍是打断体验最顺的一档。

| 参数 | 作用 |
|---|---|
| `--audio-source mic\|file:PATH` | 音频来源；`file:` = 文件当麦克风（无需硬件） |
| `--voice-rounds N` / `--voice-loop` | 跑几轮就退出 / 文件源循环（多轮回归） |
| `--voice-sink speaker\|null` | 真播放 / 只走流程不出声 |
| `--barge-ms N` | 连续说话多久算插话（默认 300ms） |
| `--stt-model NAME` / `--voice-phase1` | whisper 模型大小 / 退回整句级老链路 |

## 十七、现场身份（#18：看得见 → 认得人 → 知道在跟谁说话）

语音轮可以看一眼摄像头、认出画面里的人，并把结果带进对话 —— 于是它会**直接叫你的名字**，而不是「画面里那位」。**默认关闭**，要显式开启。

```bash
pip install "handcraft-agent[vision]"      # opencv-python<5 + numpy
python main.py --voice --identify          # 约 1Hz 识别在场的人
```

**整条链**：摄像头看到脸 → 与本地身份库比对 → 每轮回答前把**一行现场描述**注入 system 提示 → 模型据此称呼在场的人。真机实测（同一帧、同一时刻）：

| 检查项 | 结果 |
|---|---|
| 登记配方 vs 识别配方（同一人同一帧）| **0.894 / 0.894** —— 向量完全一致（余弦 1.000）|
| 库里两人时，跨人相似度 | 中位 0.149、**最高 0.267** |
| 同一人内部最低 | **0.482** |
| 阈值 | **0.36** —— 稳落两者之间，两侧都有余量 |
| 实测误认次数 | **0**（921 帧另一人的脸去认，全部如实回「未知」）|

- **分不清就直说** ✓：画面里有两个人时，嘴动信号**分不出谁在说话**（分布重叠），它会如实说「分不清」，不硬指。「我不猜 —— 猜错比说不知道更糟」是设计要的行为，不是待修的缺陷
- **登记自己人** ✓：`--identify` 下让它给你登记 ✓。它会**跨约 5 秒采 12 帧**（让姿态有变化 ✓）—— 独立姿态留出法实测：**12 帧 94%**、3 帧只有 68% ✓；**任何帧数下都不会认错** ✓，失败形态只有「说不清」✓
- **删除某人** ✓：`face_forget 名字` 即可（连同其向量一起删 ✓）
- **隐私** ✓：只存 **128 维特征向量**，**不存任何图像** ✓；库在仓库之外 `~/.forge/faces.db` ✓；摄像头**只在 `--identify` 运行时打开** ✓，探针每次读完即关闭 ✓

**几条使用小提醒**

- `--identify` **不问就不开** ✓ —— 不开摄像头、不识别、不改动对话内容 ✓
- 外放（扬声器）时它是**半双工**：它说话时闭麦 ✓ —— **说完等它答完再开口** 最顺 ✓；想插话就用 `--ptt`（按住空格说，松开就是一句 ✓）
- 语音识别默认 Whisper `base`，**中文偏弱** ✓ —— 加 `--stt-model small` 明显更准（首次下载约 460MB ✓）
- 它分得清三种情况 ✓：「现场没人」✓ / 「有人但我不认识」✓ / 「这是满仓」✓ —— **只有第三种才会报名字** ✓

---



## 十八、测试与压测
```bash
python test_m0.py                    # M0 最小循环验收
python test_m1.py                    # M1 工程底盘五件套
python test_m2.py                    # M2 多智能体验收
python test_stress.py                # M1 压测
python test_stress_hard.py           # 残酷压测（注入/并发/长时/异常/死循环）
python test_stress_hell.py           # 地狱压测（降级/高并发/Fuzz/断网/极小预算/写冲突/50 轮长跑）
python test_stress_ultra.py          # 超级压测（100 并发/双故障/编码/超长返回/1000fuzz/100 轮）
python test_stress_diverse.py        # 多任务混跑变体（--rounds N，--net 开联网任务）
python test_structured.py            # 结构化输出
python test_context_mgmt.py          # 上下文管理（滚动摘要/裁剪）
python test_trace.py                 # trace
python test_approval.py              # 审批层
python test_knowledge.py             # 知识库
python test_error_resilience.py      # 模型调用失败兜底（402 等不崩 REPL）
python test_circuit_breaker.py       # #5 熔断：三态机 + chat/stream_chat 集成 + CLI
python test_eval.py                  # #13 评估：黄金集加载/判定/报告（全 mock）
python test_web.py                   # #12 Web：HTTP 服务端到端（起真实服务 + mock Agent）
python test_server.py                # #14 API 服务：鉴权/限流/会话持久化/端点（全 mock）
python test_sandbox_logging.py       # #4 沙箱 + #7 完整日志（假 docker / 脱敏 / 轮转）
python test_interrupt.py             # 生成期打断/引导：poll_key 跨平台 + 流式中断重生成
```

`stress_m2.py --tier 1` 可跑端到端回归（Tier1 全链路 PASS）。累计压测揪出并修复 10 个 bug（详见「实施状态与改动历史」第六节）。

---

## 十九、设计脉络（模块 → 原理）

每个模块对应一套可讲清的 Agent 原理，方便按图索骥：

- 核心循环：Agent 架构与规划执行循环、完整实现骨架
- 工具：工具调用与 MCP 协议、工具安全与沙箱（审批层）
- 上下文：上下文工程与 token 管理、节省 token 的主流方案与原理（滚动摘要 / 工具输出裁剪）
- 错误处理：错误处理与回退（重试 / 降级 / 熔断规划）
- 可观测：可观测性与成本（状态栏 / trace / token）
- 多模型：多模型路由与自由选模型讨论（角色×模型 / 辩论异构）
- 结构化 handoff：多角色多模型协作
- 里程碑方案、实施状态与改动历史、测试计划见项目配套文档
