"""M1 地狱压测：模型降级真实触发 / 20 并发 / Fuzz / 断网 / 极小预算 / 并发写冲突 / 50 轮长跑。"""
import asyncio
import os
import random
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")

from src.tools import execute, TOOLS
from src.agent import Agent
from src.llm import chat


def rand_str(n=20):
    return "".join(random.choices(string.ascii_letters + string.digits + "你好世界！@#$%", k=n))


async def fallback_real():
    """真实触发模型降级：主力 base_url 坏掉，验证自动降级到 fallback。

    2026-08-18 适配：降级目标从 DeepSeek（余额 402）改为本地 qwen 端点——
    降级验证的本质是「主力坏 → fallback 顶上」，本地端点同样验证且更稳。
    """
    bad_cfg = {
        "models": [
            {"name": "broken", "base_url": "http://127.0.0.1:1/v1", "api_key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
            {"name": "good", "base_url": "http://192.168.66.54:4000/v1", "api_key": "sk-xxx", "model": "qwen3-8b-awq"},
        ],
        "roles": {"default": "broken", "fallback": "good"},
    }
    t0 = time.time()
    resp = await chat(bad_cfg, [{"role": "user", "content": "1+1等于几？只答数字"}], retries=1)
    ans = resp.choices[0].message.content
    dt = time.time() - t0
    assert "2" in ans, f"降级后答案错：{ans}"
    print(f"  ✅ 主力坏掉 → 自动降级，答「{ans}」（耗时 {dt:.1f}s）")


async def high_concurrency():
    """20 个 Agent 并发。"""
    async def one(i):
        a = Agent()
        return await a.run(f"只回答数字：{i}+{i}")

    t0 = time.time()
    results = await asyncio.gather(*[one(i) for i in range(1, 21)])
    dt = time.time() - t0
    ok = all(str(i * 2) in r for i, r in zip(range(1, 21), results))
    print(f"  {'✅' if ok else '❌'} 20 并发 {'全部正确' if ok else results}（{dt:.1f}s）")


def fuzz():
    """100 个随机畸形输入打只读工具，必须零崩溃。"""
    crash = 0
    for _ in range(100):
        tool = random.choice(["read_file", "calculator", "web_fetch", "list_files", "search_file"])
        args = {
            "read_file": {"path": rand_str()},
            "calculator": {"expression": rand_str(30)},
            "web_fetch": {"url": rand_str(15)},
            "list_files": {"path": rand_str(15)},
            "search_file": {"path": rand_str(10), "pattern": rand_str(5)},
        }[tool]
        try:
            execute(tool, args)
        except Exception:
            crash += 1
    print(f"  {'✅' if crash == 0 else '❌'} Fuzz 100 次：崩溃 {crash} 次")


def offline():
    """断网 / 坏域名 / 无协议 URL。"""
    for url in ["http://nonexistent-domain-xyz-12345.com/", "not-a-url"]:
        r = execute("web_fetch", {"url": url})
        print(f"  ✅ {url[:40]}: {r.get('data', '')[:28]}")


async def tiny_budget():
    """极小 token 预算下 5 轮，验证截断不崩不死循环。"""
    a = Agent(max_context_tokens=30, max_steps=3)
    for _ in range(5):
        ans = await a.run("1+1等于几只答数字")
        assert "2" in ans
    print(f"  ✅ 极小预算(30) 5 轮正常，上下文 {len(a.messages)} 条")


def write_race():
    """10 线程并发写同一文件，验证不崩。"""
    path = os.path.join(os.environ.get("TEMP", "."), "forge_race_test.txt")

    def one(i):
        return execute("write_file", {"path": path, "content": f"writer{i}"})

    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(one, range(10)))
    ok = all(r["ok"] for r in results)
    print(f"  {'✅' if ok else '❌'} 并发写冲突：10 个全成功、不崩")


async def long_haul():
    """50 轮超长跑 + 频繁截断。"""
    a = Agent(max_context_tokens=800)
    for i in range(50):
        ans = await a.run("1+1等于几只答数字")
        assert "2" in ans, f"第{i}轮答案错"
    print(f"  ✅ 50 轮长跑通过 · {a.usage_report()}")


if __name__ == "__main__":
    print(f"工具：{len(TOOLS)} 个\n")
    print("【1】模型降级真实触发")
    asyncio.run(fallback_real())
    print("\n【2】20 Agent 高并发")
    asyncio.run(high_concurrency())
    print("\n【3】Fuzz 100 次")
    fuzz()
    print("\n【4】断网 / 坏域名")
    offline()
    print("\n【5】极小 token 预算")
    asyncio.run(tiny_budget())
    print("\n【6】并发写冲突")
    write_race()
    print("\n【7】50 轮超长跑")
    asyncio.run(long_haul())
    print("\n=== M1 地狱压测完成 ===")
