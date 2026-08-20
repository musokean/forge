"""#5 熔断真实冒烟：坏 default 端点 + 好 fallback 指本地 qwen，验证熔断真实生效。

构造一个 default 指向连不上的端点（127.0.0.1:1）、fallback 指向本地 qwen 的配置，
反复调用 chat()，预期：
  1) 前几次 default 失败但没到阈值 → 每次都降级到本地 qwen 成功；
  2) 连续失败达阈值(2) → default 熔断 OPEN；
  3) 熔断后调用直接跳过 default，瞬间走 fallback，不再傻等重试。
运行：python smoke_circuit.py
"""
import asyncio
import sys

sys.path.insert(0, ".")

from src.llm import chat
from src.circuit import get_circuit_registry, reset_circuit_registry

CFG = {
    "roles": {"default": {"model": "bad"}, "fallback": {"model": "good"}},
    "models": {
        "bad": {"base_url": "http://127.0.0.1:1/v1", "api_key": "x", "model": "m"},
        "good": {
            "base_url": "http://192.168.66.54:4000/v1",
            "api_key": "sk-xxx",
            "model": "qwen3-8b-awq",
        },
    },
    "circuit_breaker": {"failure_threshold": 2, "cooldown": 30},
}


async def main():
    reset_circuit_registry()
    for i in range(1, 5):
        try:
            r = await chat(CFG, [{"role": "user", "content": "只说数字：1+1等于几"}], role="default", retries=1)
            ans = r.choices[0].message.content
            print(f"第{i}次  ✅ 返回：{ans[:30]!r}")
        except Exception as e:
            print(f"第{i}次  ❌ {type(e).__name__}: {e}")
        reg = get_circuit_registry(CFG)
        b = reg.get("default")
        print(f"        default 状态={b.state} · 连续失败={b.failures}")


if __name__ == "__main__":
    asyncio.run(main())
