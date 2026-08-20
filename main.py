"""CLI 入口（对应 A19 界面，命令行）。

安装后直接敲 `forge` 启动：
  forge                # 进入交互式对话（一直聊，/exit 退出）
  forge "你的问题"      # 单次问答
"""
import asyncio
import datetime
import os
import sys
import time

from src.agent import Agent
from src.orchestrator import debate, run_parallel, get_debate_roles
from src.router import route
from src.config import load_config, resolve_model
from src.tools import _get_kb, reset_kb
from src.tasks import get_scheduler
from src.config_writer import (
    available_model_aliases,
    available_roles,
    config_path,
    set_debate_model,
    set_debate_rounds,
    set_kb_path,
    set_model_key,
    set_role_model,
    set_router_role,
    setup_debate_defaults,
)
from src.console import C, paint, rule, full_rule, display_width, pad_display

NAME = "forge"  # 命令名（想改名字：改这里 + pyproject.toml 的 [project.scripts]）
# ============ ASCII 大字（ANSI Shadow 风格）============

_LETTERS = {
    "F": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "██║     ", "╚═╝     "],
    "O": [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "R": ["██████╗  ", "██╔══██╗ ", "██████╔╝ ", "██╔══██╗ ", "██║  ██║ ", "╚═╝  ╚═╝ "],
    "G": [" ██████╗ ", "██╔════╝ ", "██║  ███╗", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
}


def _big_text(text: str) -> str:
    lines = ["" for _ in range(6)]
    for ch in text:
        art = _LETTERS.get(ch)
        if not art:
            continue
        for i in range(6):
            lines[i] += art[i] + " "
    return "\n".join(l.rstrip() for l in lines)


def _model_row(label: str, model: str, width: int = 10) -> str:
    """对齐展示「角色 → 模型」一行。"""
    return "  " + pad_display(label, width) + model


def _welcome() -> str:
    cfg = load_config()
    L = []  # 收集渲染行

    # —— 大 logo（ASCII 大字，浅蓝粗体）——
    L.append(paint(_big_text("FORGE"), C.LIGHT_BLUE + C.BOLD))
    L.append(paint("把想法锻造成现实 · Forging ideas into action", C.SKY_DIM))
    L.append("")

    # —— 主力模型（只留最关键的：default 角色）——
    try:
        m = resolve_model(cfg, "default")
        label = m.get("label") or "主力"
        L.append("  " + paint(f"{label}", C.BOLD) + paint("  " + m["model"], C.SKY))
        L.append("")
    except Exception:
        pass

    # —— 一句话入口（全部引导进 /help，欢迎页保持简洁）——
    L.append(paint("  直接提问即可对话 · /help 查看全部命令 / 工具 / 快捷键", C.SKY_DIM))

    return "\n".join(L)


def _help_text() -> str:
    cfg = load_config()
    L = []
    # —— 角色 / 辩论（动态读配置）——
    try:
        L.append("■ 角色模型")
        for role in cfg.get("roles", {}):
            m = resolve_model(cfg, role)
            label = m.get("label") or role
            L.append(f"  {label} → {m['model']}")
        debate_roles, rounds = get_debate_roles(cfg)
        if debate_roles:
            L.append(f"■ 辩论阵容（{rounds} 轮）")
            for r in debate_roles:
                m = resolve_model(cfg, r["model"])
                L.append(f"  {r['name']} → {m['model']}")
        L.append("")
    except Exception:
        pass
    L += [
        "■ 工具",
        "  文件   read_file · write_file · edit_file · list_files · search_file",
        "  联网   web_search · web_fetch",
        "  知识库 kb_search · kb_ingest · kb_add",
        "  其他   calculator · run_command · get_time",
        "",
        "■ 命令",
        "/reset  清空对话上下文\n"
        "/usage  查看 token 用量\n"
        "/trace  查看本次会话的步骤流水（模型步 / 工具调用 / 耗时）\n"
        "/kb     知识库管理：add 标题|内容 · list · search 词 · ingest 路径 · sync [路径] · export [标题] · delete 标题 · path 新库路径\n"
        "/export 导出当前对话为 Markdown（/export 文件名.md）\n"
        "/key    配 key 三用法：/key sk-xxx 直接贴给主力 · /key 别名 [key] 指定模型 · /key 进向导\n"
        "/model  一键切主模型（/model 模型别名，立即生效）\n"
        "/config 配置中心（改角色模型 / 辩论阵容 / 轮数 / 路由，改完即时生效）\n"
        "/circuit 熔断状态查看 / 复位：/circuit · /circuit reset · /circuit reset <角色>\n"
        "/skill   技能包切换：/skill · /skill on <名称> · /skill off <名称>\n"
        "/memory  长期记忆管理：/memory list · forget <关键词> · clear · stats\n"
        "/remember 显式记一条关于你的记忆（如 /remember 我是做跨境电商的）\n"
        "/task   自动任务（定时/周期执行）：\n"
        "         /task                          列出全部自动任务\n"
        "         /task add <名> <调度> [--kb] <提示词>   登记（示例：每2小时 / 每天09:00 / once 2026-08-20T14:00）\n"
        "         /task del <名>                删除任务\n"
        "         /task run <名>               立即手动跑一次\n"
        "         /task on <名> / off <名>      启用 / 停用\n"
        "         /task log [<名>]             查看执行记录\n"
        "         /task clear                  清空执行记录\n"
        "/eval   黄金集回归（防变笨，A10）：\n"
        "         /eval                         跑全量黄金集（关键词命中 + LLM-as-judge 双通道）\n"
        "         /eval list                    列出黄金集用例\n"
        "         /eval add 任务|关键词1,关键词2|最低分   新增用例\n"
        "         /eval <序号>                  只跑单个用例\n"
        "         /eval export                  导出回归报告为 Markdown\n"
        "/web    Web 网页界面：#12 的浏览器入口\n"
        "         /web                         拉起 Web 聊天界面（浏览器访问，/web stop 停止）\n"
        "         或在启动时直接：forge --web [--port 8000]\n"
        "/exit   退出（或直接输 exit / quit）\n",
        "",
        "■ 生成中",
        "  按 Esc 立即中断；按任意键输入引导（如「简洁点」「换个角度」）让 forge 重新生成",
        "",
        "■ 配 key 最短路径",
        "  /key sk-xxx             直接贴 key → 自动配给当前主力模型",
        "  /key                    交互向导：选模型 → 贴 key → 可选一键切主力",
        "",
        "直接输入问题即可对话；多任务 / 需要辩论时 forge 会自动判断并分派",
    ]
    return "\n".join(L)


def _ok(ok: bool, msg: str) -> None:
    print(paint(("✅ " if ok else "❌ ") + msg, C.SKY if ok else C.RED))


def _pick(prompt: str, options, labeler) -> str | None:
    """列出可选项让用户选序号（也接受直接输名字）；返回选中的名字，取消返回 None。"""
    for k, name in enumerate(options, 1):
        print(paint(f"    {k}. ", C.LIGHT_BLUE) + labeler(name))
    v = input(paint(prompt, C.SKY_DIM)).strip()
    if not v:
        return None
    if v.isdigit() and 1 <= int(v) <= len(options):
        return options[int(v) - 1]
    return v if v in options else None


def _pick_model(prompt: str, aliases, cfg) -> str | None:
    """选模型；预置/未配 key 的模型选中后引导输入 key（明文或 env:环境变量名）。

    返回就绪的模型别名；取消/输入无效返回 None。
    """
    alias = _pick(prompt, aliases,
                  lambda a: f"{a}  ({resolve_model(cfg, a).get('label') or ''} · {resolve_model(cfg, a)['model']})")
    if not alias:
        return None
    m = resolve_model(cfg, alias)
    if m.get("api_key"):
        return alias  # 已有 key（字面或环境变量），直接可用
    print(paint(f"  「{m.get('label') or alias}」还没配 key：", C.SKY))
    v = input(paint("  输入 API Key（明文，如 sk-xxx）；或输 env:环境变量名 走环境变量 › ", C.SKY_DIM)).strip()
    if not v:
        print(paint("  已取消（未配置 key）", C.SKY_DIM))
        return None
    if v.startswith("env:"):
        env = v[4:].strip()
        if not env:
            print(paint("  env: 后要跟环境变量名，如 env:DEEPSEEK_API_KEY", C.RED))
            return None
        ok, msg = set_model_key(alias, env, use_env=True)
    else:
        ok, msg = set_model_key(alias, v)
    _ok(ok, msg)
    return alias if ok else None


def _config_panel() -> bool:
    """交互式配置中心：全程选序号，不用记语法、不用手改 yaml。返回是否有改动。"""
    changed = False
    while True:
        cfg = load_config()
        role_keys = available_roles(cfg)
        aliases = available_model_aliases(cfg)
        debate_roles, rounds = get_debate_roles(cfg)
        router_role = (cfg.get("router") or {}).get("role", "—")

        print()
        print(paint("⚙ forge 配置中心", C.LIGHT_BLUE + C.BOLD) + paint("  改完立即生效，无需重启", C.SKY_DIM))
        print(rule())
        print(paint("■ 角色 → 模型", C.BOLD))
        for i, r in enumerate(role_keys, 1):
            m = resolve_model(cfg, r)
            label = m.get("label") or r
            print(paint(f"  [{i}] ", C.LIGHT_BLUE) + pad_display(label, 10) + pad_display(r, 11)
                  + paint("→ ", C.SKY_DIM) + pad_display(m["name"], 16) + paint(m["model"], C.SKY_DIM))
        base = len(role_keys)
        print(paint(f"■ 辩论阵容（轮数 {rounds}）", C.BOLD))
        if not debate_roles:
            print(paint("  未配置。按 d 一键配置默认阵容（正方/反方/裁判）", C.SKY_DIM))
        else:
            for j, dr in enumerate(debate_roles, 1):
                try:
                    real = resolve_model(cfg, dr["model"])["model"]
                except ValueError:
                    real = paint("⚠ 角色不存在", C.RED)
                print(paint(f"  [{base + j}] ", C.LIGHT_BLUE) + pad_display(dr["name"], 10)
                      + pad_display(dr["model"], 11) + paint("→ ", C.SKY_DIM) + paint(real, C.SKY_DIM))
        print(paint("■ 可用模型", C.BOLD))
        custom_a = [a for a in aliases if not resolve_model(cfg, a).get("preset")]
        preset_a = [a for a in aliases if resolve_model(cfg, a).get("preset")]

        def _alias_line(a):
            m = resolve_model(cfg, a)
            key_state = "已配 key" if m.get("api_key") else paint("缺 key", C.RED)
            return ("      " + pad_display(a, 20) + pad_display(m.get("label") or "", 24)
                    + paint(f"{m['model']} · {key_state}", C.SKY_DIM))

        if custom_a:
            print(paint("  自建：", C.SKY_DIM))
            for a in custom_a:
                print(_alias_line(a))
        if preset_a:
            print(paint("  预置主流（缺 key 的选中后引导输入）：", C.SKY_DIM))
            for a in preset_a:
                print(_alias_line(a))
        print(paint(f"■ 自动路由判断角色：{router_role}", C.SKY_DIM))
        print(rule())
        print(paint("编号 改模型 · d 一键配辩论 · r 改轮数 · t 改路由角色 · f 打开配置文件路径 · q 返回", C.SKY_DIM))

        try:
            sel = input(paint("配置 › ", C.LIGHT_BLUE)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if sel in ("q", "", "exit", "quit", "/exit"):
            break

        if sel == "d":
            ok, msg = setup_debate_defaults()
            _ok(ok, msg)
            changed = changed or ok
            continue
        if sel == "f":
            print(paint("  " + config_path(), C.SKY_DIM))
            continue
        if sel == "r":
            v = input(paint(f"新轮数（当前 {rounds}，1~10）› ", C.SKY_DIM)).strip()
            ok, msg = set_debate_rounds(v)
            _ok(ok, msg)
            changed = changed or ok
            continue
        if sel == "t":
            role = _pick("路由判断用哪个角色 › ", role_keys,
                         lambda r: f"{r}  ({resolve_model(cfg, r).get('label') or ''} · {resolve_model(cfg, r)['model']})")
            if not role:
                print(paint("  取消", C.SKY_DIM))
                continue
            ok, msg = set_router_role(role)
            _ok(ok, msg)
            changed = changed or ok
            continue

        if not sel.isdigit():
            print(paint("  没看懂：输编号 / r / t / f / q", C.SKY_DIM))
            continue

        n = int(sel)
        if 1 <= n <= base:
            role = role_keys[n - 1]
            print(paint(f"给角色「{role}」换模型：", C.BOLD))
            alias = _pick_model(f"选模型编号（或直接输别名）› ", aliases, cfg)
            if not alias:
                print(paint("  取消", C.SKY_DIM))
                continue
            ok, msg = set_role_model(role, alias)
            _ok(ok, msg)
            changed = changed or ok
        elif base < n <= base + len(debate_roles):
            dr = debate_roles[n - base - 1]
            print(paint(f"辩手「{dr['name']}」绑到哪个角色（角色会带出对应模型）：", C.BOLD))
            role = _pick(f"选角色编号（或直接输角色名）› ", role_keys,
                         lambda r: f"{r}  ({resolve_model(cfg, r).get('label') or ''} · {resolve_model(cfg, r)['model']})")
            if not role:
                print(paint("  取消", C.SKY_DIM))
                continue
            ok, msg = set_debate_model(dr["name"], role)
            _ok(ok, msg)
            changed = changed or ok
        else:
            print(paint("  编号超出范围", C.SKY_DIM))
    return changed


async def _run_once(task: str) -> None:
    agent = Agent()
    print(full_rule())
    print(paint("任务", C.SKY_BOLD) + paint(" › ", C.SKY_DIM) + task)
    print(full_rule())
    result = await agent.run(task)  # 流式打印思考+答案
    print()
    print(paint(agent.usage_report(), C.SKY_DIM))
    print(full_rule())


def _auto_sync_kb() -> None:
    """启动时静默同步已登记的知识库目录（/kb sync 登记过的一键重放）。"""
    kb = _get_kb()
    for d in kb.sync_dirs():
        if not os.path.exists(d):
            continue
        a, u, r = kb.sync(d)
        if a or u or r:
            print(paint(f"  📚 知识库已同步「{d}」：+{a} 新增 · ~{u} 更新 · -{r} 清理", C.SKY_DIM))


def _key_command(arg: str) -> None:
    """给模型配 key，三种用法（全部免记长命令）：
      /key sk-xxx              直接贴 key → 自动配给当前主力模型（default 角色），最常用
      /key <模型别名> sk-xxx    指定模型配 key（别名输不全时自动补全/提示）
      /key                     交互向导：列模型 → 选 → 贴 key → 可选一键切主力
    也支持 env:环境变量名（如 /key env:DEEPSEEK_API_KEY）。
    """
    parts = arg.split(None, 1)
    if not parts:
        _key_wizard()
        return
    first = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""

    # 情况一：直接给的是 key（sk- 或 env:）→ 自动配给 default 角色的当前模型
    if first.startswith("sk-") or first.startswith("env:"):
        cfg = load_config()
        try:
            cur = resolve_model(cfg, "default")
            alias = cur["name"]
        except Exception:
            print(paint("  ⚠ 解析当前主力模型失败，请用 /key 模型别名 sk-xxx", C.RED))
            return
        value = first + (" " + rest if rest else "") if not first.startswith("env:") else first
        if first.startswith("env:"):
            ok, msg = set_model_key(alias, first[4:], use_env=True)
        else:
            ok, msg = set_model_key(alias, value.strip())
        _ok(ok, f"{msg}（当前主力「{alias}」）")
        print(paint(f"  提示：下次启动或 /model {alias} 即可生效（当前会话如需立即切换：/model {alias}）", C.SKY_DIM))
        return

    # 情况二：给了模型别名
    alias = first
    cfg = load_config()
    aliases = available_model_aliases(cfg)
    if alias not in aliases:
        # 模糊匹配：前缀命中直接补全
        hits = [a for a in aliases if a.startswith(alias)]
        if len(hits) == 1:
            alias = hits[0]
            print(paint(f"  已补全模型名：{alias}", C.SKY_DIM))
        else:
            print(paint(f"  模型「{alias}」不存在，可用：{', '.join(aliases[:12])}" + (" …" if len(aliases) > 12 else ""), C.RED))
            print(paint("  也可以直接 /key sk-xxx 自动配给当前主力，或 /key 进向导", C.SKY_DIM))
            return
    value = rest
    if not value:
        value = input(paint(f"  给「{alias}」输入 API Key（sk-…，或 env:环境变量名）› ", C.SKY_DIM)).strip()
    if not value:
        print(paint("  已取消", C.SKY_DIM))
        return
    if value.startswith("env:"):
        ok, msg = set_model_key(alias, value[4:], use_env=True)
    else:
        ok, msg = set_model_key(alias, value)
    _ok(ok, msg)
    if ok and value.startswith("sk-"):
        print(paint("  提示：/model 别名 可把主模型切到这个模型，立即生效", C.SKY_DIM))


def _key_wizard() -> None:
    """/key 交互向导：列模型 → 选序号 → 贴 key → 可选切主力。"""
    cfg = load_config()
    aliases = available_model_aliases(cfg)
    if not aliases:
        print(paint("  models.yaml 里没有可用模型", C.RED))
        return
    try:
        cur = resolve_model(cfg, "default")["name"]
    except Exception:
        cur = "?"
    print(paint(f"  给哪个模型配 key？（当前主力：{cur}）", C.BOLD))
    custom = [a for a in aliases if not resolve_model(cfg, a).get("preset")]
    preset = [a for a in aliases if resolve_model(cfg, a).get("preset")]
    n = 0
    for group in (custom, preset):
        for a in group:
            n += 1
            m = resolve_model(cfg, a)
            ks = "✅" if m.get("api_key") else "缺"
            star = " ★" if a == cur else ""
            print(paint(f"  [{n}] ", C.LIGHT_BLUE) + pad_display(a, 22)
                  + paint(f"{m.get('label') or ''} · {m['model']}", C.SKY_DIM)
                  + paint(f"  {ks}{star}", C.SKY if m.get('api_key') else C.RED))
    print(paint("  输序号选模型，回车取消；也可直接贴 sk-xxx 配给主力", C.SKY_DIM))
    try:
        v = input(paint("  模型编号或 key › ", C.LIGHT_BLUE)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not v:
        return
    if v.isdigit() and 1 <= int(v) <= len(aliases):
        alias = aliases[int(v) - 1]
        m = resolve_model(cfg, alias)
        key = input(paint(f"  给「{alias}」输入 API Key（sk-… 或 env:变量名）› ", C.SKY_DIM)).strip()
        if not key:
            print(paint("  已取消", C.SKY_DIM))
            return
        if key.startswith("env:"):
            ok, msg = set_model_key(alias, key[4:], use_env=True)
        else:
            ok, msg = set_model_key(alias, key)
        _ok(ok, msg)
        if ok and key.startswith("sk-"):
            sw = input(paint(f"  要把主力切成「{alias}」吗？[y/N] ", C.SKY_DIM)).strip().lower()
            if sw in ("y", "yes"):
                ok2, msg2 = set_role_model("default", alias)
                _ok(ok2, msg2)
        return
    # 直接贴了 key
    if v.startswith("sk-") or v.startswith("env:"):
        _key_command(v)
        return
    print(paint("  没看懂：输编号或直接贴 sk-xxx", C.SKY_DIM))


def _check_keys(cfg) -> None:
    """启动时检测常用角色的模型是否缺 key，缺则提醒（模型管理流程优化）。"""
    miss = []
    for role in ("default", "reasoning", "fallback", "chinese"):
        try:
            m = resolve_model(cfg, role)
            if not m.get("api_key"):
                miss.append(f"{role}({m.get('model')})")
        except Exception:
            continue
    if miss:
        print(paint(f"  ⚠ 以下角色缺可用 key：{'、'.join(miss)}", C.SKY))
        print(paint("  配 key：/key 模型别名 sk-xxx · 或 /config 面板选模型引导输入", C.SKY_DIM))


async def _probe_endpoint(cfg, role="default", timeout=5.0) -> tuple:
    """启动自检：探测主力端点连通性（5s 硬超时，不阻塞启动）。

    返回 (ok: bool, detail: str)。直连 _call_role 不走降级链——探测的是 default 本身。
    """
    from src.llm import _call_role
    m = resolve_model(cfg, role)
    try:
        resp = await asyncio.wait_for(
            _call_role(cfg, [{"role": "user", "content": "回复 OK 两个字"}], role, retries=1),
            timeout=timeout,
        )
        return True, f"{m['model']} @ {m['base_url']}（连通 ✅）"
    except asyncio.TimeoutError:
        return False, f"{m['model']} @ {m['base_url']}（超时 {timeout}s，端点不可达）"
    except Exception as e:
        return False, f"{m['model']} @ {m['base_url']}（{type(e).__name__}: {str(e)[:60]}）"


def _kb_command(rest: str) -> None:
    """知识库管理：#15 的 CLI 管理入口。用法：/kb（状态）· /kb ingest 路径 · /kb search 词 · /kb sync [路径] · /kb path 新库路径。"""
    kb = _get_kb()
    if not rest:
        st = kb.stats()
        dirs = kb.sync_dirs()
        print(paint(f"■ 知识库（{kb.db_path}）", C.BOLD))
        print(paint(f"  文档 {st['docs']} 篇（库内条目 {st['inline']} · 文件 {st['docs'] - st['inline']}）· 累计 {st['chars']:,} 字符", C.SKY_DIM))
        if dirs:
            print(paint(f"  自动同步目录（启动时重放）：{len(dirs)} 个", C.SKY_DIM))
        print(paint("  用法：/kb add 标题|内容 · list · search 词 · ingest 路径 · sync [路径] · export [标题] · delete 标题 · path 新库路径", C.SKY_DIM))
        return
    cmd, _, arg = rest.partition(" ")
    arg = arg.strip()
    if cmd == "path" and arg:
        ok, msg = set_kb_path(arg)
        _ok(ok, msg)
        if ok:
            reset_kb()  # 单例失效，下次 _get_kb 按新路径重建
            print(paint(f"  新库路径：{_get_kb().db_path}", C.SKY_DIM))
        return
    if cmd == "add":
        if "|" in arg:
            title, content = [s.strip() for s in arg.split("|", 1)]
            _ok(*kb.add(title, content))
        elif arg:
            print(paint("  用法：/kb add 标题|内容（用 | 分隔）", C.SKY_DIM))
        else:
            title = input(paint("  标题 › ", C.SKY_DIM)).strip()
            content = input(paint("  内容 › ", C.SKY_DIM)).strip()
            if title and content:
                _ok(*kb.add(title, content))
            else:
                print(paint("  已取消（标题和内容都不能为空）", C.SKY_DIM))
        return
    if cmd == "list":
        entries = kb.list_entries()
        if not entries:
            print(paint("  库内还没有条目：/kb add 标题|内容 沉淀第一条", C.SKY_DIM))
            return
        print(paint(f"■ 库内条目（{len(entries)} 条）", C.BOLD))
        for i, e in enumerate(entries, 1):
            stamp = datetime.datetime.fromtimestamp(e["updated"]).strftime("%m-%d %H:%M")
            print(paint(f"  {i}. ", C.LIGHT_BLUE) + pad_display(e["title"], 24)
                  + paint(f"{e['chars']}字 · {stamp}", C.SKY_DIM))
        return
    if cmd == "delete":
        if not arg:
            print(paint("  用法：/kb delete 标题（或 /kb list 里的编号）", C.SKY_DIM))
            return
        if arg.isdigit():
            entries = kb.list_entries()
            n = int(arg)
            if 1 <= n <= len(entries):
                arg = entries[n - 1]["title"]
            else:
                print(paint(f"  编号超出范围（共 {len(entries)} 条）", C.RED))
                return
        _ok(*kb.delete(arg))
        return
    if cmd == "export":
        out_dir = os.path.join(EXPORT_DIR, "kb")
        exported = kb.export_entries(out_dir, arg or None)
        if not exported:
            print(paint("  没有可导出的条目", C.SKY_DIM))
            return
        print(paint(f"✅ 已导出 {len(exported)} 条 → {out_dir}", C.SKY))
        for p in exported:
            print(paint(f"  {os.path.basename(p)}", C.SKY_DIM))
        return
    if cmd == "sync":
        targets = [arg] if arg else kb.sync_dirs()
        if not targets:
            print(paint("  没有登记过同步目录：/kb sync <路径> 登记并同步", C.SKY_DIM))
            return
        total = (0, 0, 0)
        for t in targets:
            if not os.path.exists(t):
                print(paint(f"  ⚠ 目录不存在，跳过：{t}", C.SKY))
                continue
            a, u, r = kb.sync(t)
            total = (total[0] + a, total[1] + u, total[2] + r)
            print(paint(f"  📚 同步「{t}」：+{a} 新增 · ~{u} 更新 · -{r} 清理", C.SKY_DIM))
        if arg:
            kb.register_sync_dir(arg)  # 登记，启动时自动重放
        print(paint(f"✅ 同步完成：+{total[0]} 新增 · ~{total[1]} 更新 · -{total[2]} 清理"
                    f"（库内共 {kb.stats()['docs']} 篇）", C.SKY))
        return
    if cmd == "ingest" and arg:
        t0 = time.monotonic()
        added, skipped = kb.ingest(arg)
        st = kb.stats()
        print(paint(f"✅ 建索引完成：新增 {added} · 跳过 {skipped} · 库内共 {st['docs']} 篇"
                    f"（{time.monotonic() - t0:.2f}s）", C.SKY))
        return
    if cmd == "search" and arg:
        hits = kb.query(arg)
        if not hits:
            print(paint(f"  检索「{arg}」：无命中", C.SKY_DIM))
            return
        print(paint(f"■ 检索「{arg}」（{len(hits)} 条）", C.BOLD))
        for i, h in enumerate(hits, 1):
            print(paint(f"  {i}. ", C.LIGHT_BLUE) + paint(h["title"], C.BOLD))
            print(paint(f"     {h['snippet']}", C.SKY_DIM))
            print(paint(f"     {h['path']}", C.SKY_DIM))
        return
    print(paint("用法：/kb（状态）· add 标题|内容 · list · search 词 · ingest 路径 · sync [路径] · export [标题] · delete 标题 · path 新库路径", C.SKY_DIM))


EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")


def _export_markdown(agent: Agent, name: str = "") -> str:
    """把当前对话导出为 Markdown 文件（exports/ 目录）。返回文件路径。"""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    if name and not name.endswith(".md"):
        name = f"{name}.md"
    fname = name or f"对话-{datetime.datetime.now():%Y%m%d-%H%M%S}.md"
    path = os.path.join(EXPORT_DIR, fname)
    lines = [f"# forge 对话导出（{agent.name} · {datetime.datetime.now():%Y-%m-%d %H:%M}）", ""]
    for m in agent.messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            continue
        if role == "user":
            if content.startswith("[早期对话摘要"):
                lines += ["> 📎 早期对话摘要", f"> {content.split(']', 1)[-1].strip()}", ""]
            else:
                lines += ["### 🧑 用户", content, ""]
        elif role == "assistant":
            if m.get("tool_calls"):
                names = [tc["function"]["name"] for tc in m["tool_calls"]]
                lines += [f"### ⚙ {agent.name}（调用工具：{'、'.join(names)}）", "", ""]
            else:
                lines += [f"### ⚙ {agent.name}", content, ""]
        elif role == "tool":
            lines += ["> 🔧 工具结果", "> ```json", "> " + content[:500], "> ```", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _circuit_command(arg: str) -> None:
    """熔断状态查看 / 手动复位：#5 熔断的 CLI 入口（对应 A07 错误隔离）。
    用法：
      /circuit              列出各角色熔断状态
      /circuit reset        复位全部熔断
      /circuit reset <角色>  复位指定角色（如 /circuit reset default）
    """
    from src.circuit import get_circuit_registry
    reg = get_circuit_registry(load_config())
    _y = getattr(C, "YELLOW", C.SKY_DIM)
    parts = arg.split()
    if parts and parts[0] == "reset":
        if len(parts) >= 2:
            reg.reset(parts[1])
            print(paint(f"  ✅ 已复位熔断：{parts[1]}", C.SKY))
        else:
            reg.reset()
            print(paint("  ✅ 已复位全部熔断", C.SKY))
        return
    snap = reg.snapshot()
    if not snap:
        print(paint("  暂无熔断记录（角色都还健康，或尚未发生过模型调用）", C.SKY_DIM))
        return
    print(paint("  熔断器状态（#5 熔断）：", C.BOLD))
    for name, s in snap.items():
        st = s["state"]
        if st == "open":
            print(paint(f"  🔴 {name}: OPEN · 冷却剩余 {s['cooldown_remaining']}s", C.RED))
        elif st == "half_open":
            print(paint(f"  🟡 {name}: HALF_OPEN · 探测中", _y))
        else:
            print(paint(f"  🟢 {name}: CLOSED · 连续失败 {s['failures']}", C.SKY))


def _skill_command(arg: str, agent: Agent) -> None:
    """技能包查看 / 切换：/skill on|off <名称>（Skill 系统 CLI 入口）。
    用法：
      /skill              列出全部技能及激活态
      /skill on <名称>    激活技能（提示词片段+工具白名单注入）
      /skill off <名称>   停用技能
    """
    from src.skills import list_skills, activate, deactivate, skill_status_line
    parts = arg.split()
    if parts and parts[0] == "on" and len(parts) >= 2:
        try:
            activate(parts[1])
            agent.refresh_system()
            print(paint(f"  ✅ 已激活技能「{parts[1]}」，系统提示已更新", C.SKY))
            return
        except ValueError as e:
            print(paint(f"  ⚠ {e}", C.RED))
            return
    if parts and parts[0] == "off" and len(parts) >= 2:
        try:
            deactivate(parts[1])
            agent.refresh_system()
            print(paint(f"  ✅ 已停用技能「{parts[1]}」，系统提示已更新", C.SKY))
            return
        except ValueError as e:
            print(paint(f"  ⚠ {e}", C.RED))
            return
    skills = list_skills()
    print(paint(f"  技能包（当前激活：{skill_status_line()}）", C.BOLD))
    for s in skills:
        mark = "🟢" if s["active"] else "⚪"
        print(paint(f"  {mark} {s['name']}：{s['description']}", C.SKY_DIM if not s["active"] else C.SKY))


def _task_command(arg: str) -> None:
    """自动任务管理（#18 定时/周期执行，对应 A28）：
      /task                             列出全部任务
      /task add <名> <调度> [--kb] <提示词>   登记（--kb 把结果沉淀进知识库）
      /task del <名>                   删除
      /task run <名>                  立即手动跑一次
      /task on <名> / off <名>        启用 / 停用
      /task log [<名>]                查看执行记录（默认最近 10 条）
      /task clear                      清空执行记录
    调度示例：每2小时 / 每30分钟 / 每天09:00 / once 2026-08-20T14:00
    """
    from src.tasks import get_scheduler
    sched = get_scheduler()
    parts = arg.split()
    sub = parts[0] if parts else ""
    if sub == "add" and len(parts) >= 4:
        name, sched_text = parts[1], parts[2]
        rest = parts[3:]
        kb_sink = "--kb" in rest
        if kb_sink:
            rest = [x for x in rest if x != "--kb"]
        prompt = " ".join(rest)
        msg = sched.add(name, sched_text, prompt, kb_sink=kb_sink)
        print(paint(msg, C.SKY if msg.startswith("✅") else C.RED))
        return
    if sub == "del" and len(parts) >= 2:
        print(paint(sched.delete(parts[1]), C.SKY))
        return
    if sub == "run" and len(parts) >= 2:
        print(paint(sched.run_now(parts[1]), C.SKY))
        return
    if sub in ("on", "off") and len(parts) >= 2:
        print(paint(sched.set_enabled(parts[1], sub == "on"), C.SKY))
        return
    if sub == "log":
        name = parts[1] if len(parts) >= 2 else None
        runs = sched.recent_runs(name, limit=10)
        if not runs:
            print(paint("  暂无执行记录。", C.SKY_DIM))
            return
        label = f"「{name}」" if name else "全部"
        print(paint(f"  自动任务执行记录（{label}，最近 {len(runs)} 条）：", C.BOLD))
        for r in runs:
            mark = "✅" if r["ok"] else "⚠"
            out = (r["output"] or r["error"] or "")[:70].replace("\n", " ")
            print(paint(f"  {mark} {r['task_name']} · {r['finished_at']} · {out}", C.SKY_DIM))
        return
    if sub == "clear":
        n = sched.clear_runs()
        print(paint(f"  ✅ 已清空 {n} 条执行记录", C.SKY))
        return
    # 无子命令 → 列表
    tasks = sched.list_tasks()
    if not tasks:
        print(paint("  暂无自动任务。登记示例：/task add 每日简报 每天09:00 帮我总结今天的重要事项", C.SKY_DIM))
        return
    print(paint("  自动任务（#18，后台周期执行；forge 运行中自动触发）：", C.BOLD))
    for t in tasks:
        mark = "🟢" if t["enabled"] else "⚪"
        extra = " →沉淀知识库" if t["kb_sink"] else ""
        nr = t["next_run"] or "—"
        print(paint(f"  {mark} {t['name']}（{t['sched_type']}:{t['expr']}）下次 {nr}{extra}", C.SKY_DIM))
        print(paint(f"      提示词：{t['prompt'][:70]}", C.SKY_DIM))


def _memory_command(arg: str) -> None:
    """长期记忆管理：/memory list · forget <关键词> · clear · stats（跨会话用户画像）。
    /remember <内容> 显式记一条。
    """
    from src.memory import get_memory
    m = get_memory()
    parts = arg.split()
    if parts and parts[0] == "forget" and len(parts) >= 2:
        n = m.forget(" ".join(parts[1:]))
        print(paint(f"  ✅ 已删除 {n} 条记忆", C.SKY if n else C.SKY_DIM))
        return
    if parts and parts[0] == "clear":
        n = m.clear()
        print(paint(f"  ✅ 已清空全部 {n} 条记忆", C.SKY))
        return
    if parts and parts[0] == "stats":
        st = m.stats()
        print(paint(f"  记忆库：{st['count']} 条 · 存储 {st['path']}", C.SKY_DIM))
        return
    mems = m.list_all()
    st = m.stats()
    print(paint(f"  长期记忆（共 {st['count']} 条，跨会话保留）", C.BOLD))
    if not mems:
        print(paint("  暂无记忆。可说「记住我是做跨境电商的」让我记住你，或 /remember <内容> 显式写入", C.SKY_DIM))
        return
    for mm in mems:
        print(paint(f"  [{mm['id']}] {mm['content']}", C.SKY_DIM) + paint(f"（{mm['hit_count']}次命中）", C.SKY_DIM))


def _eval_command(arg: str) -> None:
    """黄金集回归（#13 评估 / A10 防变笨）：
      /eval              跑全量黄金集回归（关键词命中 + LLM-as-judge 双通道判定）
      /eval list         列出黄金集用例
      /eval add 任务|关键词1,关键词2|最低分   新增用例（写回 config/golden.yaml）
      /eval <序号>       只跑第 N 个用例
      /eval export       把最近一次回归结果导出为 Markdown（exports/eval/）
    """
    from src.eval import Evaluator
    global _EVAL_LAST_REPORT
    ev = Evaluator()
    parts = arg.split()
    sub = parts[0] if parts else ""

    if sub == "list":
        cases = ev.load_golden()
        print(paint(f"  黄金集（{len(cases)} 例，config/golden.yaml）：", C.BOLD))
        for i, c in enumerate(cases, 1):
            kw = "、".join(c.get("keywords") or []) or "无"
            note = f"  # {c['note']}" if c.get("note") else ""
            print(paint(f"  [{i}] ", C.LIGHT_BLUE) + pad_display(c["task"][:40], 42)
                  + paint(f"关键词:{kw} · ≥{c.get('min_score', 6)}分", C.SKY_DIM) + paint(note, C.SKY_DIM))
        print(paint("  跑回归：/eval · 单例：/eval <序号> · 新增：/eval add 任务|关键词1,关键词2|最低分", C.SKY_DIM))
        return

    if sub == "add":
        rest = arg[3:].strip()
        if "|" not in rest:
            print(paint("  用法：/eval add 任务|关键词1,关键词2|最低分（如：/eval add 计算 12×12|144|6）", C.SKY_DIM))
            return
        task, _, kw_part = rest.partition("|")
        kw_text, _, score_text = kw_part.partition("|")
        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
        try:
            min_score = int(score_text.strip() or 6)
        except ValueError:
            min_score = 6
        ok, msg = ev.add_case(task.strip(), keywords, min_score)
        _ok(ok, msg)
        return

    if sub == "export":
        rep = _EVAL_LAST_REPORT
        if not rep:
            print(paint("  还没有回归结果可导出：先跑一次 /eval", C.SKY_DIM))
            return
        path = ev.export_markdown(rep)
        print(paint(f"✅ 已导出回归报告：{path}", C.SKY))
        return

    if sub.isdigit():
        cases = ev.load_golden()
        n = int(sub)
        if not (1 <= n <= len(cases)):
            print(paint(f"  序号超出范围（共 {len(cases)} 例）", C.RED))
            return
        print(paint(f"🧪 跑黄金用例 [{n}]：{cases[n - 1]['task']}", C.LIGHT_BLUE + C.BOLD))
        result = asyncio.run(ev.run_case(cases[n - 1]))
        _EVAL_LAST_REPORT = [result]
        print(ev.report([result]))
        return

    # 默认：全量回归
    cases = ev.load_golden()
    print(paint(f"🧪 黄金集回归开始（{len(cases)} 例，并发跑批）…", C.LIGHT_BLUE + C.BOLD))
    t0 = time.monotonic()
    results = asyncio.run(ev.run_all(cases))
    _EVAL_LAST_REPORT = results
    print(ev.report(results, title=f"⏱ 总耗时 {time.monotonic() - t0:.1f}s"))
    passed = sum(1 for r in results if r.passed)
    if passed == len(results):
        print(paint("  ✅ 全部通过：forge 核心能力未见退化", C.SKY))
    else:
        print(paint("  ⚠ 有未通过的用例：/eval export 导出详情，检查是能力退化还是用例过时", C.SKY))


# 最近一次 /eval 回归结果（供 /eval export 导出）
_EVAL_LAST_REPORT = None


async def _handle_line(agent: Agent, cfg, line: str) -> None:
    """处理一行普通输入：自动路由 → 并行 / 辩论 / 直答。异常由 _repl 兜底（不崩）。"""
    # 长期记忆双钩子：先自动沉淀（「我喜欢/我是…」模式），再召回相关记忆注入上下文
    from src.memory import get_memory
    mem = get_memory()
    mem.auto_remember(line)
    mem_ctx = mem.compose_context(line)
    if mem_ctx:
        line = mem_ctx + "\n\n用户问题：" + line
    from src.router import _rule_route, route
    from src.spinner import spinner_start, spinner_stop
    # 规则预判命中：0ms 秒回，不显示 spinner（老大 2026-08-20：规则命中还闪「判断任务类型」很烦）
    decision = _rule_route(line)
    if decision is None:
        # 只有规则拿不准的才启动 spinner 走模型兜底
        spin = spinner_start("🧭 判断任务类型")
        try:
            decision = await route(line, cfg)
        finally:
            spinner_stop(spin)
    if decision["type"] == "parallel" and decision.get("subtasks"):
        subs = decision["subtasks"]
        print(paint(f"🧩 识别为多任务，并行拆解 {len(subs)} 个子任务…", C.BOLD))
        results = await run_parallel(subs)
        for t, r in zip(subs, results):
            print(paint(f"任务「{t}」", C.SKY) + paint(" → ", C.SKY_DIM) + r[:80])
        print(full_rule())
        return
    if decision["type"] == "debate":
        from src.orchestrator import get_debate_roles
        dr, _ = get_debate_roles(cfg)
        if not dr:
            # 辩论阵容未配置：提示并降级普通直答（老大 2026-08-20：辩论默认不配，有需要再配）
            print(paint("⚖️ 这是个决策问题。当前未配置辩论阵容，按普通问答回答；", C.SKY)
                  + paint("需要多角色辩论可 /config 按 d 一键配置", C.SKY_DIM))
            result = await agent.run(line)
            if result and not result.endswith(("（已中断）", "（已连续打断，中止）")):
                if result.startswith(("⚠", "达到最大步数", "（")):
                    print(paint(result, C.YELLOW if not result.startswith("⚠") else C.RED))
            print(full_rule())
            return
        q = decision.get("question") or line
        print(paint("⚖️ 识别为决策问题，多角色辩论…", C.BOLD))
        await debate(q)  # 各角色+裁判流式发言
        print(full_rule())
        return
    if decision["type"] == "plan":
        from src.orchestrator import run_supervised
        ans = await run_supervised(line)  # planner 拆解 → 并行执行 → merger 合并
        print(ans)
        print(full_rule())
        return
    result = await agent.run(line)  # 流式打印思考+答案
    # 流式路径答案已打印；但失败/中断等非流式返回需显式输出（老大 2026-08-20：失败被吞只看到分隔线）
    if result and not result.endswith(("（已中断）", "（已连续打断，中止）")):
        # agent.run 流式时已打印内容（返回值和打印内容一致），这里只补打非流式/失败提示
        if result.startswith(("⚠", "达到最大步数", "（")):
            print(paint(result, C.SKY if not result.startswith("⚠") else C.RED))
    print(full_rule())


def _web_command(arg: str) -> None:
    """Web 界面管理：#12 的 REPL 入口。
    /web          拉起 Web 聊天界面（后台线程跑 HTTP 服务，浏览器访问）
    /web stop     停止 Web 服务
    """
    global _WEB_INSTANCE
    arg = arg.strip()
    if arg == "stop":
        if _WEB_INSTANCE:
            _WEB_INSTANCE.stop()
            _WEB_INSTANCE = None
        else:
            print(paint("  Web 服务未在运行", C.SKY_DIM))
        return
    if _WEB_INSTANCE:
        print(paint(f"  Web 已在运行：{_WEB_INSTANCE.url}", C.SKY_DIM))
        return
    from src.web import ForgeWeb
    _WEB_INSTANCE = ForgeWeb(auto_open=False)
    _WEB_INSTANCE.start()
    print(paint(f"  （浏览器访问 {_WEB_INSTANCE.url}；/web stop 停止）", C.SKY_DIM))


# 全局 Web 实例（/web 命令管理）
_WEB_INSTANCE = None


async def _repl() -> None:
    agent = Agent()  # 复用同一个实例，保留多轮上下文
    cfg = load_config()
    print(_welcome())
    _check_keys(cfg)  # 启动缺 key 检测（模型管理流程优化）
    # 启动自检：探测主力端点连通性（老大 2026-08-20：端点不可达时曾「卡住没回复」）
    try:
        ok, detail = await _probe_endpoint(cfg)
        print(paint(("  ✅ 主力端点 " if ok else "  ❌ 主力端点 ") + detail,
                    C.SKY if ok else C.RED))
        if not ok:
            print(paint("     （可 /model 切换其他模型，或检查网络/端点/余额；问题仍会尽力用降级通道回答）", C.SKY_DIM))
    except Exception:
        pass  # 自检失败不阻塞启动
    _auto_sync_kb()  # 启动时静默同步已登记的知识库目录
    sched = get_scheduler()
    sched.start()    # 自动任务调度器（#18）：后台线程周期执行登记的任务
    # —— 介绍区（欢迎页/提示）与会话区用两行贯穿全宽的分隔线分开（老大 2026-08-20：一直到底）——
    print()
    print(full_rule())
    print(full_rule())
    print()
    while True:
        try:
            line = input("你 › ").strip()
        except (EOFError, KeyboardInterrupt):
            print(paint("\n" + agent.usage_report(), C.SKY_DIM))
            print("再见。")
            sched.stop()
            break
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit"):
            print(paint(agent.usage_report(), C.SKY_DIM))
            print(paint("再见。", C.SKY_DIM))
            sched.stop()
            break
        if line in ("/reset", "reset"):
            agent.reset()
            print(paint("（对话上下文已重置）", C.SKY_DIM))
            continue
        if line in ("/usage", "usage"):
            print(paint(agent.usage_report(), C.SKY_DIM))
            continue
        if line in ("/trace", "trace"):
            print(paint(agent.trace_report(), C.SKY_DIM))
            continue
        if line == "/kb" or line.startswith("/kb "):
            _kb_command(line[3:].strip())
            continue
        if line == "/export" or line.startswith("/export "):
            path = _export_markdown(agent, line[8:].strip())
            print(paint(f"✅ 已导出 Markdown：{path}", C.SKY))
            continue
        if line in ("/help", "help", "?"):
            print(paint(_help_text(), C.SKY_DIM))
            continue
        if line == "/key" or line.startswith("/key "):
            _key_command(line[5:].strip())
            continue
        if line == "/model" or line.startswith("/model "):
            arg = line[7:].strip()
            if not arg:
                cfg_now = load_config()
                aliases = available_model_aliases(cfg_now)
                print(paint("  用法：/model 模型别名 —— 把主模型（default 角色）切到指定模型，立即生效", C.SKY_DIM))
                print(paint("  可用：", C.SKY_DIM) + " · ".join(aliases[:10]) + (" …" if len(aliases) > 10 else ""))
                continue
            ok, msg = set_role_model("default", arg)
            _ok(ok, msg)
            if ok:
                cfg = load_config()          # 刷新自动路由用配置
                agent.reload_config()        # 主对话模型热重载（保留上下文）
                print(paint("（主模型已切换，立即生效）", C.SKY))
            continue
        if line in ("/config", "/models", "config"):
            if _config_panel():
                cfg = load_config()          # 自动路由用的配置
                agent.reload_config()        # 主对话模型（保留上下文）
                print(paint("（配置已热重载，当前对话立即生效）", C.SKY))
            continue
        if line == "/circuit" or line.startswith("/circuit "):
            _circuit_command(line[9:].strip())
            continue
        if line == "/skill" or line.startswith("/skill "):
            _skill_command(line[6:].strip(), agent)
            continue
        if line == "/memory" or line.startswith("/memory "):
            _memory_command(line[7:].strip())
            continue
        if line == "/remember" or line.startswith("/remember "):
            from src.memory import get_memory
            ok, msg = get_memory().remember(line[9:].strip())
            _ok(ok, msg)
            continue
        if line == "/task" or line.startswith("/task "):
            _task_command(line[5:].strip())
            continue
        if line == "/eval" or line.startswith("/eval "):
            _eval_command(line[5:].strip())
            continue
        if line == "/web" or line.startswith("/web "):
            _web_command(line[4:].strip())
            continue
        # 自动路由：AI 自己判断任务类型（单答 / 并行 / 辩论）
        try:
            await _handle_line(agent, cfg, line)
        except Exception as e:
            print(paint(f"⚠ 出错了：{e}", C.RED))
            print(paint("  （已接住异常，对话继续；可用 /trace 查看流水）", C.SKY_DIM))
            continue


def main() -> None:
    # 启动参数：
    #   forge --web [--port 8000]   起 Web 聊天界面（零依赖 HTTP 服务，浏览器访问）
    #   forge "问题"                 单次问答
    #   forge                       交互式对话
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        port = 8000
        if "--port" in sys.argv:
            try:
                port = int(sys.argv[sys.argv.index("--port") + 1])
            except (ValueError, IndexError):
                pass
        from src.web import start_web
        web = start_web(port=port)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            web.stop()
        return
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        asyncio.run(_run_once(task))
    else:
        asyncio.run(_repl())


if __name__ == "__main__":
    main()
