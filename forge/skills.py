"""技能包系统（Skill）：预置「提示词片段 + 工具子集」按需装配。

与主流 Agent 的 Skill/插件概念对应：把某类任务所需的行为约束与工具集合打包，
按需激活，避免把所有提示塞进上下文（省 token）+ 避免无关工具干扰模型判断。

用法：
    from .skills import activate, deactivate, compose_prompt, schema_filter
    activate("coding")                       # 开启技能
    compose_prompt(base_system)              # 基础提示词 + 激活技能片段
    schema_filter(full_schema)               # 只保留激活技能白名单内的工具
"""
from .config import load_config

# 技能注册表：name -> {"description", "prompt", "tools"(可选工具白名单)}
SKILLS = {
    "coding": {
        "description": "编程助手：写代码、修 bug、读代码、改项目。激活后优先用 read_file/search_file 先看代码再动手，改动前先列清单。",
        "prompt": (
            "你现在启用了「编程助手」技能："
            "写代码/修 bug 前先用 read_file / search_file / list_files 看清楚相关代码再动手，不要凭猜测改；"
            "修改用 edit_file 做局部替换而非整文件重写；改动前先用一句话说明你的改动计划。"
        ),
        "tools": ["read_file", "write_file", "edit_file", "list_files", "search_file", "run_command"],
    },
    "writing": {
        "description": "文档写作：写报告、方案、邮件、知识条目。强调结构清晰、分节标题、结论先行。",
        "prompt": (
            "你现在启用了「文档写作」技能："
            "输出用清晰的分节结构（标题+要点+表格），结论先行再展开论据；"
            "面向中文读者，术语首次出现给中文解释；写完检查一遍逻辑连贯性再交付。"
        ),
        "tools": None,
    },
    "research": {
        "description": "联网调研：查最新信息、对比资料、查证事实。强调多角度搜索、引用来源。",
        "prompt": (
            "你现在启用了「联网调研」技能："
            "用 web_search 至少搜 1-2 次覆盖不同关键词，再用 web_fetch 打开 1-2 个权威来源核实细节；"
            "回答时标注信息来源；信息冲突时如实说明，不要编造。"
        ),
        "tools": ["web_search", "web_fetch"],
    },
    "knowledge": {
        "description": "知识库助手：侧重本地知识库检索与沉淀，查资料先 kb_search，重要结论主动 kb_add。",
        "prompt": (
            "你现在启用了「知识库助手」技能："
            "涉及本地资料/历史文档的问题先用 kb_search 检索，不要凭印象回答；"
            "对话中产生值得沉淀的结论（规则、方案、踩坑）时，主动用 kb_add 写入知识库。"
        ),
        "tools": ["kb_search", "kb_ingest", "kb_add"],
    },
}

_active = set()  # 当前激活的技能名集合（运行态，随进程存活）


def list_skills():
    """返回技能元信息列表（含激活态），供 /skill list 使用。"""
    return [{"name": n, "description": s["description"], "active": n in _active}
            for n, s in SKILLS.items()]


def activate(name):
    """激活技能；未知技能抛 ValueError。"""
    if name not in SKILLS:
        raise ValueError(f"未知技能 {name}，可用：{', '.join(SKILLS)}")
    _active.add(name)
    return True


def deactivate(name):
    """停用技能；未知技能抛 ValueError。"""
    if name not in SKILLS:
        raise ValueError(f"未知技能 {name}，可用：{', '.join(SKILLS)}")
    _active.discard(name)
    return True


def active_skills():
    """当前激活的技能名列表（按注册顺序）。"""
    return [n for n in SKILLS if n in _active]


def is_active(name):
    return name in _active


def compose_prompt(base: str) -> str:
    """基础系统提示词 + 激活技能的行为片段。无激活技能时原样返回 base。"""
    if not _active:
        return base
    parts = [base]
    for n in active_skills():
        p = SKILLS[n].get("prompt")
        if p:
            parts.append(p)
    return "\n".join(parts)


def schema_filter(full_schema):
    """只保留激活技能工具白名单内的工具 schema；无激活技能时原样返回。"""
    if not _active:
        return full_schema
    whitelist = set()
    for n in active_skills():
        for t in (SKILLS[n].get("tools") or []):
            whitelist.add(t)
    if not whitelist:
        return full_schema  # 技能未声明白名单：不裁剪
    return [s for s in full_schema if s["function"]["name"] in whitelist]


def skill_status_line():
    """一行式状态摘要（欢迎页 / /skill 用）。"""
    acts = active_skills()
    return "、".join(acts) if acts else "（无）"
