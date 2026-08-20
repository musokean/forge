"""M2 渐进压测驱动器（供自动任务调用）。

用法：
  python stress_m2.py --auto           # 按当前时间距 2026-08-18 08:00 的剩余小时数自动选档(1~5)
  python stress_m2.py --tier N          # 手动指定档位 1~5

强度随档位递增（并行任务数 / 辩论轮数 / 自动路由样例数 / 多轮对话轮数）：
  T1: p3  d1  route3  round10
  T2: p5  d2  route5  round10
  T3: p8  d3  route8  round15
  T4: p12 d4  route10 round20
  T5: p15 d5  route12 round30
每档都先跑「多模型路由静态校验」（不调 LLM），再跑真实多智能体压测。
"""
import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")

from src.config import load_config, resolve_model
from src.orchestrator import run_parallel, debate, get_debate_roles
from src.router import route
from src.agent import Agent

# 截止时间：2026-08-18 08:00 Asia/Shanghai (UTC+8)
DEADLINE = datetime(2026, 8, 18, 8, 0, 0, tzinfo=timezone(timedelta(hours=8)))

SH = "Asia/Shanghai"


def remaining_hours() -> float:
    now = datetime.now(timezone(timedelta(hours=8)))
    return max(0.0, (DEADLINE - now).total_seconds() / 3600.0)


def auto_tier() -> int:
    h = remaining_hours()
    if h >= 5:
        return 1
    if h >= 4:
        return 2
    if h >= 3:
        return 3
    if h >= 2:
        return 4
    return 5


# 样例任务库（覆盖 single / parallel / debate 三类，供自动路由校验）
SAMPLE_TASKS = [
    ("single", "1+1 等于几？只答数字"),
    ("parallel", "帮我并行做三件事：算 12*12、查一下今天北京的天气、用一句话解释什么是向量数据库"),
    ("debate", "跨境电商团队该不该用 AI 全自动写开发信？请权衡利弊"),
    ("single", "今天农历几月几号？"),
    ("parallel", "并行：把『知识管理』翻译成英文、算 99+1、列出 3 个主流开源 LLM 框架"),
    ("debate", "小团队要不要自建模型微调，还是直接调 API？"),
    ("single", "用一句话说明 ReAct 范式"),
    ("parallel", "并行处理：算 100/4、解释什么是 MCP、用中文写一句欢迎语"),
    ("debate", "CLI 工具和 Web 界面哪个更适合 Agent 产品？"),
    ("single", "Python 里 list 和 tuple 的区别是什么？"),
    ("parallel", "并行：算 7*6、查一下上海今天气温、解释什么是 RAG"),
    ("debate", "Agent 应该优先追求能力强，还是优先追求可控安全？"),
]


def check_routing_static():
    """不调 LLM：检查辩论角色绑定的模型（A17 多模型路由）。
    本地单模型部署时 distinct=1 属预期，仅作告警，不判失败。"""
    cfg = load_config()
    debate_roles, _ = get_debate_roles(cfg)
    models = [resolve_model(cfg, r["model"])["model"] for r in debate_roles]
    distinct = len(set(models))
    return distinct, models


async def step_parallel(n, concurrency):
    tasks = [f"只答数字：{i}+{i} 等于几" for i in range(1, n + 1)]
    t0 = time.time()
    results = await run_parallel(tasks, concurrency=concurrency)
    dt = time.time() - t0
    ok = sum(1 for i, r in enumerate(results, 1) if str(i * 2) in r)
    return ok, n, dt


async def step_debate(rounds):
    q = "Agent 产品应该先把『能力强』做到极致，还是先把『可控安全』做扎实？"
    t0 = time.time()
    ans = await debate(q, rounds=rounds)
    dt = time.time() - t0
    return bool(ans), len(ans), dt


async def step_route(k):
    cfg = load_config()
    samples = SAMPLE_TASKS[:k]
    t0 = time.time()
    types = []
    for label, task in samples:
        d = await route(task, cfg)
        types.append(d.get("type"))
    dt = time.time() - t0
    valid = sum(1 for t in types if t in ("single", "parallel", "debate"))
    # 诚实性检测：route() 在 API 报错时会静默兜底成 single。
    # 如果样例里明明有 parallel/debate，却全部被归为 single，基本可判定是 API 失败兜底。
    non_single = [lbl for lbl, _ in samples if lbl != "single"]
    all_fell_back = bool(non_single) and all(t == "single" for t in types)
    return valid, k, dt, types, all_fell_back


async def step_multi_round(n):
    a = Agent()
    t0 = time.time()
    for i in range(n):
        await a.run("1+1 等于几？只答数字")
    dt = time.time() - t0
    return n, dt, len(a.messages)


async def run_tier(tier: int):
    spec = {
        1: dict(p=3, d=1, route=3, round=10, conc=5),
        2: dict(p=5, d=2, route=5, round=10, conc=5),
        3: dict(p=8, d=3, route=8, round=15, conc=6),
        4: dict(p=12, d=4, route=10, round=20, conc=8),
        5: dict(p=15, d=5, route=12, round=30, conc=10),
    }[tier]

    print(f"[M2 STRESS] tier={tier} 剩余={remaining_hours():.1f}h spec={spec}")
    fails = []

    # 0) 静态多模型路由校验（不调 LLM）
    try:
        distinct, models = check_routing_static()
        if distinct >= 2:
            print(f"  ✅ 多模型路由静态校验：{distinct} 个不同模型 {models}")
        else:
            print(f"  ⚠️ 多模型路由静态校验：仅 {distinct} 个模型 {models}（本地单模型部署，辩论非异构，属预期）")
    except Exception as e:
        print(f"  ❌ 多模型路由静态校验失败：{e}")
        fails.append("routing_static")

    # 1) 并行执行
    try:
        ok, n, dt = await step_parallel(spec["p"], spec["conc"])
        mark = "✅" if ok == n else "⚠️"
        print(f"  {mark} 并行执行 {n} 任务：{ok}/{n} 正确（{dt:.1f}s）")
        if ok != n:
            fails.append("parallel")
    except Exception as e:
        print(f"  ❌ 并行执行崩溃：{type(e).__name__}: {e}")
        fails.append("parallel")

    # 2) 多角色辩论
    try:
        ok, length, dt = await step_debate(spec["d"])
        mark = "✅" if ok else "❌"
        print(f"  {mark} 辩论 {spec['d']} 轮：产出 {length} 字结论（{dt:.1f}s）")
        if not ok:
            fails.append("debate")
    except Exception as e:
        print(f"  ❌ 辩论崩溃：{type(e).__name__}: {e}")
        fails.append("debate")

    # 3) 自动路由校验
    try:
        valid, k, dt, types, fell_back = await step_route(spec["route"])
        mark = "✅" if (valid == k and not fell_back) else "⚠️"
        print(f"  {mark} 自动路由 {k} 样例：{valid}/{k} 正常（{dt:.1f}s） types={types}")
        if fell_back:
            print(f"  ⚠️ 自动路由疑似 API 失败兜底（parallel/debate 样例被全部归为 single）")
            fails.append("route(fallback)")
        elif valid != k:
            fails.append("route")
    except Exception as e:
        print(f"  ❌ 自动路由崩溃：{type(e).__name__}: {e}")
        fails.append("route")

    # 4) 多轮对话
    try:
        n, dt, msgs = await step_multi_round(spec["round"])
        print(f"  ✅ 多轮对话 {n} 轮：上下文 {msgs} 条（{dt:.1f}s）")
    except Exception as e:
        print(f"  ❌ 多轮对话崩溃：{type(e).__name__}: {e}")
        fails.append("multi_round")

    total = "FAIL" if fails else "PASS"
    print(f"[M2 STRESS] tier={tier} 结果={total} 失败项={fails or '无'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--auto", action="store_true")
    args = ap.parse_args()
    tier = args.tier or (auto_tier() if args.auto else auto_tier())
    asyncio.run(run_tier(tier))


if __name__ == "__main__":
    main()
