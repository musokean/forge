"""编排层：并行执行 + 讨论式多智能体 + 多模型路由 + supervisor 规划执行（对应 A15/A16/A17/A27）。"""
import asyncio

from pydantic import BaseModel, Field

from .agent import Agent
from .config import load_config
from .approval import Approver
from .structured import RoleBrief, ask_structured
from .llm import chat

# 默认讨论角色（仅当 config/models.yaml 的 debate.roles 缺失时才用，作为兜底）
DEFAULT_DEBATE_ROLES = [
    {"name": "正方", "model": "reasoning", "persona": "你代表正方立场，坚定论证这个观点/方案的合理性，找出并强调它的优点与价值。"},
    {"name": "反方", "model": "fallback", "persona": "你代表反方立场，质疑并找出这个观点/方案的漏洞、风险与代价，客观反驳。"},
    {"name": "裁判", "model": "default", "persona": "你是中立的裁判，不偏袒任何一方，综合双方观点，给出客观、平衡、可执行的最终结论。"},
]


def get_debate_roles(cfg=None):
    """从配置读取辩论阵容与轮数；配置缺失（没有 debate 段/没写 roles 键）时回退 DEFAULT_DEBATE_ROLES。

    注意（老大 2026-08-20）：配置里显式 `roles: []` 表示「默认不配辩论、有需要再配」，
    必须返回空列表——`[]` 是 falsy，不能和「缺失」混为一谈（否则 or 回退默认阵容就白配了）。
    """
    if cfg is None:
        cfg = load_config()
    d = (cfg or {}).get("debate", {}) or {}
    roles = d.get("roles", None)
    if roles is None:  # 只有「没写 roles 键」才回退默认阵容
        roles = DEFAULT_DEBATE_ROLES
    rounds = d.get("rounds", 2)
    return roles, rounds


async def run_parallel(tasks, concurrency=5, structured=False, trace_summary=True):
    """并行执行多个独立任务（A15），信号量限流防止撞上游 rate limit。

    structured=True 时，每个任务返回校验过的 RoleBrief 对象（A27 结构化 handoff），
    而不是散文——下游拿结构化字段，不用解析上游文本。
    trace_summary=True 时，结束后每任务打印一行 trace 摘要（#7 多智能体可观测）。
    """
    sem = asyncio.Semaphore(concurrency)
    agents = []

    async def one(t):
        async with sem:
            # 无人值守批量：写操作自动放行（审批留给交互式主对话）；不显示 spinner（并发多行会刷屏）
            a = Agent(stream=False, approver=Approver(auto_approve=True), show_spinner=False)
            agents.append(a)
            if structured:
                return await a.run_structured(t, RoleBrief)
            return await a.run(t)

    results = await asyncio.gather(*[one(t) for t in tasks])
    if trace_summary:
        from .console import C, paint
        for a in agents:
            print(paint(f"  ⛓ {a.tracer.one_line()}", C.DIM))
    return results


async def debate(question, roles=None, rounds=None, judge_structured=False):
    """讨论式多智能体（A16 + A17）：多角色辩论 + 裁判汇总。

    roles: [{"name", "model", "persona"}, ...]，name 含「裁判」的作为最终裁判（其余为辩手）。
           不传则从 config/models.yaml 的 debate 段读取（配置驱动）。
    rounds: 辩论轮数，不传则用配置 debate.rounds（默认 2）。
    judge_structured: True 时裁判返回校验过的 RoleBrief 对象（A27 结构化 handoff），否则返回散文。
    """
    if roles is None:
        roles, cfg_rounds = get_debate_roles()
        rounds = rounds if rounds is not None else cfg_rounds
    rounds = rounds or 2
    debaters = [r for r in roles if "裁判" not in r["name"]]
    judges = [r for r in roles if "裁判" in r["name"]]
    judge = judges[0] if judges else None

    # 每个角色独立 Agent：独立上下文 + 绑不同模型（A17 多模型路由），流式发言带角色名前缀
    # 辩论角色默认放行写操作（无人值守；审批留给交互式主对话）；不显示 spinner（多角色轮流发言会刷屏）
    agents = [(r["name"], Agent(role=r["model"], system_prompt=r["persona"], name=r["name"],
                                approver=Approver(auto_approve=True), show_spinner=False)) for r in debaters]

    transcript = []
    # 第一轮：各角色发表初始观点
    for name, a in agents:
        ans = await a.run(f"请就以下问题发表你的看法：\n{question}")
        transcript.append({"name": name, "round": 1, "text": ans})

    # 后续轮：看到他人观点后回应
    for rnd in range(2, rounds + 1):
        for name, a in agents:
            others = [t for t in transcript if t["name"] != name]
            others_text = "\n".join(f"[{t['name']}] {t['text']}" for t in others[-len(agents):])
            ans = await a.run(f"其他角色的观点：\n{others_text}\n\n请回应并更新你的观点：")
            transcript.append({"name": name, "round": rnd, "text": ans})

    # 裁判汇总
    if judge:
        j = Agent(role=judge["model"], system_prompt=judge["persona"], name=judge["name"],
                  approver=Approver(auto_approve=True), show_spinner=False)
        full = "\n".join(f"[{t['name']} 第{t['round']}轮] {t['text']}" for t in transcript)
        if judge_structured:
            return await j.run_structured(
                f"以下是各方讨论记录：\n{full}\n\n请给出最终结论的要点摘要。",
                RoleBrief,
            )
        return await j.run(f"以下是各方讨论记录：\n{full}\n\n请给出最终结论。")
    return transcript[-1]["text"]


# ============ supervisor：规划 → 并行执行 → 合并（router 从「只分类」升级的分派能力） ============


class PlanItem(BaseModel):
    """supervisor 拆解出的单个子任务（指令自包含，执行者不依赖其他上下文）。"""

    title: str = Field(description="子任务标题（一句话）")
    instruction: str = Field(description="子任务详细指令（自包含，执行者只看到它即可完成）")


class Plan(BaseModel):
    """supervisor 的整体拆解结果。"""

    plan_title: str = Field(description="整个任务的执行计划标题")
    subtasks: list[PlanItem] = Field(description="子任务列表（2-4 个，相互独立可并行）")


async def run_supervised(task, planner_role="reasoning", merger_role="default", max_subtasks=4):
    """supervisor 流水：planner 拆解 → 并行执行（复用 #8）→ merger 合并。

    任一环节失败自动降级：planner 拆解失败 → 退化为普通单 Agent 直答；
    部分子任务失败 → 其余结果照常合并（失败子任务带错误说明，不整体崩）。
    返回最终合并答案（str）。
    """
    cfg = load_config()
    from .structured import StructuredError
    from .console import C, paint

    # 1) planner 拆解（reasoning 角色，结构化输出保证可解析）
    print("🧭 supervisor：正在拆解任务为子任务…")
    from .spinner import spinner_start, spinner_stop
    spin = spinner_start("🧭 拆解任务")
    try:
        plan, _ = await ask_structured(
            cfg,
            [{"role": "user", "content": (
                "你是任务规划师（supervisor 的 planner）。把下面的用户任务拆成 2-4 个相互独立、"
                "可并行执行的子任务。规则：子任务指令必须自包含（执行者只看到它就能独立完成，"
                "不要引用其他子任务的产出）；不要拆分出「总结/合并」类子任务（那是 merger 的活）。\n\n"
                f"任务：{task}"
            )}],
            Plan,
            role=planner_role,
        )
        subtasks = [s.instruction for s in plan.subtasks][:max_subtasks]
        spinner_stop(spin)
    except (StructuredError, Exception):
        # 拆解失败：退化为普通直答（plan 是增强路径，不是必由之路）
        spinner_stop(spin)
        a = Agent(stream=False, approver=Approver(auto_approve=True), show_spinner=False)
        return await a.run(task)

    # 2) 并行执行（结构化 handoff：每个子任务产出 RoleBrief，失败子任务在合并时显式标注）
    print(paint(f"🧭 已拆解 {len(subtasks)} 个子任务，并行执行…", C.BOLD))
    briefs = await run_parallel(subtasks, structured=True, trace_summary=True)

    # 3) merger 合并（default 角色综合成面向用户的最终答案）
    parts = []
    for t, b in zip(subtasks, briefs):
        if isinstance(b, RoleBrief):
            pts = "；".join(b.key_points) if b.key_points else "（无要点）"
            parts.append(f"[子任务：{t[:50]}] 结论：{b.stance or '—'}\n要点：{pts}")
        else:
            parts.append(f"[子任务：{t[:50]}] 执行失败：{str(b)[:200]}")
    merged = "\n\n".join(parts)
    print("🧭 正在合并各子任务结果…")
    spin2 = spinner_start("🧭 合并结果")
    msgs = [
        {"role": "system", "content": (
            "你是 forge 的总编辑。把多个子任务的结果合并成一份连贯、完整、面向最终用户的最终答案："
            "直接回答用户原始问题，子任务结论不足的部分如实说明，不要编造。"
        )},
        {"role": "user", "content": f"原始用户任务：{task}\n\n各子任务产出：\n{merged}"},
    ]
    try:
        resp = await chat(cfg, msgs, role=merger_role)
        spinner_stop(spin2)
        return resp.choices[0].message.content or ""
    except Exception as e:
        spinner_stop(spin2)
        return f"⚠ 合并子任务结果失败：{e}"
