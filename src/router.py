"""任务路由：自动判断任务类型（单答 / 并行拆解 / 多角色辩论），路由到对应执行方式。

策略（老大 2026-08-20 两次反馈「🧭 判断任务类型卡住」）：
  1. **规则预判优先**：问候/简单问答/明显的辩论/多任务/规划任务用正则零成本秒判，
     不调模型——「你好」「今天天气」这类直接 single 秒回；
  2. **模型兜底**：只有规则拿不准的才调模型判断，且：
     · 硬超时 ROUTE_TIMEOUT=5s（路由只是「先猜一下」，超时直接降级 single，绝不阻塞用户）
     · 不走降级链（直接 _call_role 单次尝试，失败立刻降级 single，不被不可达端点拖住）
  路由是增强路径，不是必由之路——猜错顶多少用并行/辩论，答案依然正确。
"""
import asyncio
import json
import re

from .llm import _call_role

# 路由判断硬超时（秒）：5s 是「还能接受」的上限，超时降级 single
ROUTE_TIMEOUT = 5

# ---------- 规则预判（零成本，秒回） ----------

# 问候 / 寒暄 → single
_GREETING = re.compile(
    r"^(?:你好|嗨|哈喽|hello|hi|在吗|在不在|谢谢|多谢|再见|拜拜|拜拜啦|ok|好的|嗯|辛苦了|早上好|晚上好|下午好)[!！。~～.\s]*$",
    re.I,
)
# 决策 / 权衡 / 观点碰撞 → debate
_DEBATE = re.compile(
    r"该不该|要不要|值不值得|怎么选|选哪个|选什么|哪个好|哪家好|还是|"
    r"利弊|利.{0,4}弊|风险.{0,6}(评估|权衡)|值得吗|划算吗|赞成|反对|支持.{0,6}还是反对|vs",
)
# 多步 / 调研 / 报告 / 规划 → plan
_PLAN = re.compile(
    r"调研|梳理|写份?报告|写个报告|整理.{0,4}方案|制定.{0,4}计划|规划|"
    r"分析.{0,4}报告|选型|建议.{0,8}(方案|怎么做)|把.{0,12}都办|分步骤|流程设计|架构设计",
)
# 明显多任务 → parallel（「分别/同时」+ 动作；「A、B、C」顿号分隔的并列动作）
_PARALLEL = re.compile(
    r"(分别|同时|逐个|依次).{0,12}(查|算|写|读|搜|总结|列|分析|整理|生成)"
    r"|(查|算|写|读|搜|总结|列|分析|整理|生成).{0,6}(和|跟|与).{0,6}(查|算|写|读|搜|总结|列|分析|整理|生成)"
    r"|(查|算|写|读|搜|总结|列|分析|整理|生成).{0,4}[、，,].{0,4}(查|算|写|读|搜|总结|列|分析|整理|生成)",
)
# 常见简单问句（无并列动作）→ single，省一次模型调用
_SIMPLE_Q = re.compile(
    r"(今天|现在|几点|什么时间|星期几|多少|等于|怎么算|计算|天气|气温|温度|"
    r"简介|是什么|是谁|在哪|在哪里|怎么做|能不能|可以.{0,4}吗|会不会|为什么|什么意思|"
    r"翻译|推荐|介绍|解释|说说|讲讲|总结下|写一段|写个|帮我写)",
)


def _rule_route(task: str):
    """规则预判：命中返回路由 dict，未命中返回 None（交给模型兜底）。"""
    t = (task or "").strip()
    if not t:
        return {"type": "single", "subtasks": [], "question": t}
    if _GREETING.match(t):
        return {"type": "single", "subtasks": [], "question": t}
    if _DEBATE.search(t):
        return {"type": "debate", "subtasks": [], "question": t}
    if _PLAN.search(t):
        return {"type": "plan", "subtasks": [], "question": t}
    if _PARALLEL.search(t):
        return {"type": "parallel", "subtasks": [], "question": t}
    if _SIMPLE_Q.search(t):
        return {"type": "single", "subtasks": [], "question": t}
    return None


async def route(task, cfg, role=None):
    """判断任务类型，返回 {"type": "single|parallel|plan|debate", "subtasks": [...], "question": "..."}。

    先规则预判（零成本秒回），规则未命中才调模型（5s 硬超时 + 无降级链）；
    一切失败兜底 single。role 不传时读配置 router.role。
    """
    # 1) 规则预判：命中直接返回，不调模型
    r = _rule_route(task)
    if r is not None:
        return r

    # 2) 模型兜底：只对「拿不准的」花一次模型调用
    if role is None:
        role = (cfg or {}).get("router", {}).get("role", "fallback")
    messages = [
        {"role": "system", "content": "你是任务路由器。判断用户任务该「单答 / 并行拆解 / 规划执行 / 多角色辩论」，只输出一个 JSON 对象，不要其他内容。"},
        {"role": "user", "content": (
            "判断下面任务的类型，输出 JSON：\n"
            '{"type": "single|parallel|plan|debate", "subtasks": ["..."], "question": "..."}\n\n'
            "判断规则：\n"
            "- single：单一问题，直接回答即可（如「帮我写一段产品介绍」「推荐几本书」）\n"
            "- parallel：包含多个互不依赖的独立子任务（如「查天气、查汇率、算个数」），subtasks 列出拆分后的各子任务\n"
            "- plan：需要先收集信息/执行多步动作才能给出综合结论的任务（如「调研 A 和 B 的差异并给出选型建议」「梳理项目结构并写份报告」「把这几件事都办了」），subtasks 留空（由 supervisor 进一步拆解）\n"
            "- debate：需要权衡利弊、风险评估、观点碰撞的决策问题（如「该不该用 AI 写开发信」「选 A 还是 B」），question 填核心问题\n\n"
            f"任务：{task}"
        )},
    ]
    try:
        # 无降级链 + 5s 硬超时：模型不可达时最多等 5 秒就降级 single（不再被端点拖死）
        resp = await asyncio.wait_for(
            _call_role(cfg, messages, role, retries=1),
            timeout=ROUTE_TIMEOUT,
        )
        content = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", content, re.S)
        d = json.loads(m.group(0)) if m else {}
        if d.get("type") in ("single", "parallel", "plan", "debate"):
            return d
    except (asyncio.TimeoutError, Exception):
        pass  # 超时 / 失败：降级 single
    return {"type": "single", "subtasks": [], "question": task}
