"""M2 验收：多模型路由 + 并行执行 + 讨论式多智能体。"""
import asyncio
import sys
import time

sys.path.insert(0, ".")

from src.agent import Agent
from src.orchestrator import run_parallel, debate, DEFAULT_DEBATE_ROLES
from src.config import load_config, resolve_model


def test_multi_model():
    """验证辩论角色绑定了不同模型（A17 多模型路由）。"""
    cfg = load_config()
    models = [resolve_model(cfg, r["model"])["model"] for r in DEFAULT_DEBATE_ROLES]
    distinct = len(set(models))
    for r, m in zip(DEFAULT_DEBATE_ROLES, models):
        print(f"  {r['name']} → 角色「{r['model']}」= {m}")
    assert distinct >= 2, "辩论至少要有两个不同模型"
    print(f"  ✅ 多模型路由：{distinct} 个不同模型参与辩论")


async def test_parallel():
    """并行执行 3 个独立任务，验证结果正确（A15）。"""
    tasks = ["2+2 等于几（只答数字）", "3+3 等于几（只答数字）", "4+4 等于几（只答数字）"]
    t0 = time.time()
    results = await run_parallel(tasks)
    dt = time.time() - t0
    ok = all(str(i * 2) in r for i, r in zip((2, 3, 4), results))
    print(f"  并行结果：{results}（{dt:.1f}s）")
    assert ok, f"并行结果错误：{results}"
    print("  ✅ 并行执行正确")


async def test_debate():
    """讨论式多智能体：决策问题多角色辩论 + 裁判汇总（A16）。"""
    ans = await debate("跨境电商团队该不该引入 AI 来写开发信？", rounds=1)
    print(f"  裁判结论：{ans[:120]}")
    assert ans
    print("  ✅ 讨论式多智能体产出结论")


async def _run_all():
    print("\n【2】并行执行")
    await test_parallel()
    print("\n【3】讨论式多智能体")
    await test_debate()


if __name__ == "__main__":
    print("【1】多模型路由")
    test_multi_model()
    asyncio.run(_run_all())  # 同一事件循环跑完，避免跨循环复用客户端
    print("\n=== M2 测试完成 ===")
