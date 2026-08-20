# forge · 把想法锻造成现实

> **English**: [README.en.md](README.en.md)

把 M0→M3 里程碑落成代码——ReAct 核心循环 + 工程底盘（工具只读分级 / 上下文截断+滚动摘要 / 错误重试+降级 / token 统计 / 每步 trace / 写操作审批）+ 多智能体编排（并行拆解 / 讨论式辩论 / 多模型路由 / 自动路由 / 流式输出）+ 结构化输出 + 本地知识库（SQLite+FTS5）+ 黄金集评估 + Web 界面 + Markdown 导出。**模型配置驱动、零外部重依赖**。

---

## 一、快速开始

```bash
# 1. 安装（项目目录下执行一次，装成系统命令）
pip install -e .

# 2. 启动交互式对话
forge

# 3. 单次问答（不进入交互模式）
forge "帮我算 (3+5)*2"

# 4. Web 网页界面（浏览器聊天，零依赖 HTTP 服务）
forge --web                  # 默认端口 8000，自动开浏览器
forge --web --port 8080      # 指定端口
```

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

**生成中打断 / 引导**（老大 2026-08-20 需求）：
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
├── src/
│   ├── config.py         # 配置加载 + resolve_model（角色/别名解析）
│   ├── config_writer.py  # 安全写回配置（/config 面板底层）
│   ├── llm.py            # openai SDK 网关 + 重试 + 降级 + 流式 + 熔断
│   ├── tools.py          # 13 工具 + @tool 注册 + 只读分级 + 知识库工具
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
| **技能包 Skill** | `src/skills.py` | 预置「提示词片段 + 工具白名单」按需装配：`coding` 编程 / `writing` 写作 / `research` 调研 / `knowledge` 知识库，激活后只给模型相关工具（省 token + 减少误调） | `/skill` |
| **长期记忆** | `src/memory.py` | 跨会话记住你是谁：说「我是/我喜欢/我习惯…」自动沉淀；每次提问自动召回相关记忆注入上下文——forge 不再是每次见面的陌生人 | `/memory` · `/remember` |
| **反思自纠错** | `src/reflect.py` | A07 组合拳末环：答案生成后评审打分，低分带意见重答（默认关，`/config` 可开） | 配置 `reflect` |
| **Supervisor 主管** | `src/orchestrator.py` | 路由从「只分类」升级「分派+合并」：复杂任务 planner 拆解 → 并行执行 → merger 合并最终答案；拆解失败自动降级直答 | 自动（路由判定 `plan`） |

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

## 十二、测试与压测

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
python test_interrupt.py             # 生成期打断/引导：poll_key 跨平台 + 流式中断重生成
```

`stress_m2.py --tier 1` 可跑端到端回归（Tier1 全链路 PASS）。累计压测揪出并修复 10 个 bug（详见「实施状态与改动历史」第六节）。

---

## 十三、设计脉络（模块 → 原理）

每个模块对应一套可讲清的 Agent 原理，方便按图索骥：

- 核心循环：Agent 架构与规划执行循环、完整实现骨架
- 工具：工具调用与 MCP 协议、工具安全与沙箱（审批层）
- 上下文：上下文工程与 token 管理、节省 token 的主流方案与原理（滚动摘要 / 工具输出裁剪）
- 错误处理：错误处理与回退（重试 / 降级 / 熔断规划）
- 可观测：可观测性与成本（状态栏 / trace / token）
- 多模型：多模型路由与自由选模型讨论（角色×模型 / 辩论异构）
- 结构化 handoff：多角色多模型协作
- 里程碑方案、实施状态与改动历史、测试计划见项目配套文档
