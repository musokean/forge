"""M2 验收：多模型路由 + 并行执行 + 讨论式多智能体。"""
import asyncio
import sys
import time

sys.path.insert(0, ".")

from src.agent import Agent
from src.orchestrator import run_parallel, debate, get_debate_roles, DEFAULT_DEBATE_ROLES
from src.config import load_config, resolve_model


def test_multi_model():
    """多模型路由（A17）——断机制，不断本机配置。

    本机 config/models.yaml 不入库、可以只配一个模型（默认配置已精简为单角色），
    「辩论必须两个不同模型」属于配置条件，而非代码要求。所以：
      ① 合成配置验证「不同角色 → 不同模型」的路由机制；
      ② 打印本机实际阵容作参考——单模型配置下辩论照常可跑，只是各角色同模型。
    """
    synth = {
        "models": {
            "cheap": {"label": "便宜模型", "base_url": "http://x/v1", "model": "small-x", "api_key": "k"},
            "strong": {"label": "强模型", "base_url": "http://x/v1", "model": "big-x", "api_key": "k"},
        },
        "roles": {"default": {"model": "cheap"}, "reasoning": {"model": "strong"}},
    }
    lineup = [{"name": "正方", "model": "default"}, {"name": "反方", "model": "reasoning"}]
    models = [resolve_model(synth, r["model"])["model"] for r in lineup]
    assert len(set(models)) == 2, f"不同角色应解析到不同模型：{models}"
    print(f"  ✅ 多模型路由机制：{lineup[0]['name']} → {models[0]} / {lineup[1]['name']} → {models[1]}")

    # 本机实际阵容（参考信息）
    cfg = load_config()
    roles, _ = get_debate_roles(cfg)
    roles = roles or DEFAULT_DEBATE_ROLES  # 显式置空 = 默认不配辩论；debate() 内部同样兜底
    live = [resolve_model(cfg, r["model"])["model"] for r in roles]
    for r, m in zip(roles, live):
        print(f"  {r['name']} → 角色「{r['model']}」= {m}")
    print(f"  本机辩论阵容：{len(set(live))} 个不同模型参与（单模型配置合法，各角色同模型）")


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
