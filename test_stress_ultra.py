"""M1 超级压测：100 并发 / 双故障降级 / 编码地狱 / 超长返回 / 1000 次 fuzz / 100 轮长跑。"""
import asyncio
import os
import random
import string
import sys
import time

sys.path.insert(0, ".")

from src.tools import execute, TOOLS
from src.agent import Agent
from src.llm import chat


def rand_str(n=20):
    return "".join(random.choices(string.ascii_letters + string.digits + "你好世界！@#$%🎯🔍", k=n))


async def ultra_concurrency():
    """100 个 Agent 并发。"""
    async def one(i):
        a = Agent()
        return await a.run(f"只回答数字：{i}+{i}")

    t0 = time.time()
    results = await asyncio.gather(*[one(i) for i in range(1, 101)])
    dt = time.time() - t0
    ok = sum(1 for i, r in zip(range(1, 101), results) if str(i * 2) in r)
    print(f"  {'✅' if ok == 100 else '⚠️'} 100 并发：{ok}/100 正确（{dt:.1f}s）")


async def double_fault():
    """双故障：主力 + fallback 都坏，验证优雅报错、不无限重试。"""
    bad_cfg = {
        "models": [
            {"name": "broken1", "base_url": "http://127.0.0.1:1/v1", "api_key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
            {"name": "broken2", "base_url": "http://127.0.0.1:2/v1", "api_key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
        ],
        "roles": {"default": "broken1", "fallback": "broken2"},
    }
    t0 = time.time()
    try:
        await chat(bad_cfg, [{"role": "user", "content": "hi"}], retries=1)
        print("  ❌ 双故障竟然成功了？")
    except Exception as e:
        dt = time.time() - t0
        print(f"  ✅ 双故障优雅报错：{type(e).__name__}（{dt:.1f}s，未死循环）")


def encoding_hell():
    """编码地狱：GBK 文件、emoji、控制字符、命令注入尝试。"""
    gbk_path = os.path.join(os.environ.get("TEMP", "."), "forge_gbk_test.txt")
    with open(gbk_path, "wb") as f:
        f.write("中文GBK测试".encode("gbk"))
    r = execute("read_file", {"path": gbk_path})
    print(f"  read_file GBK文件: {'✅ 不崩(返回错误)' if not r['ok'] else '⚠️ 读出:' + str(r['data'])[:30]}")
    for tool, args in [
        ("web_search", {"query": "🎯🔍🚀\x00\x01\x02"}),
        ("calculator", {"expression": "1+1\n;rm -rf /"}),
        ("list_files", {"path": "🎯\x00\x01"}),
        ("run_command", {"command": "echo $(whoami)"}),
    ]:
        try:
            execute(tool, args)
            print(f"  ✅ {tool} 编码/注入地狱不崩")
        except Exception as e:
            print(f"  ❌ {tool} 崩了: {e}")


def huge_return():
    """超长工具返回：read_file 读 ~3MB 大文件，检查是否限制大小。"""
    big_path = os.path.join(os.environ.get("TEMP", "."), "forge_big_test.txt")
    with open(big_path, "w", encoding="utf-8") as f:
        f.write("大数据测试\n" * 500000)
    r = execute("read_file", {"path": big_path})
    size = len(r.get("data", "")) if r["ok"] else 0
    print(f"  read_file 大文件：返回 {size} 字符（{'⚠️ 无限制，会爆上下文' if size > 100000 else '✅ 有截断'}）")


def fuzz_1000():
    """1000 次随机 fuzz。"""
    crash = 0
    tools = ["read_file", "calculator", "web_fetch", "list_files", "search_file", "get_time"]
    for _ in range(1000):
        tool = random.choice(tools)
        args = {
            "read_file": {"path": rand_str()},
            "calculator": {"expression": rand_str(30)},
            "web_fetch": {"url": rand_str(15)},
            "list_files": {"path": rand_str(15)},
            "search_file": {"path": rand_str(10), "pattern": rand_str(5)},
            "get_time": {"x": rand_str(5)},
        }[tool]
        try:
            execute(tool, args)
        except Exception:
            crash += 1
    print(f"  {'✅' if crash == 0 else '❌'} Fuzz 1000 次：崩溃 {crash} 次")


async def long_haul():
    """100 轮长跑。"""
    a = Agent(max_context_tokens=800)
    for i in range(100):
        ans = await a.run("1+1等于几只答数字")
        assert "2" in ans, f"第{i}轮崩"
    print(f"  ✅ 100 轮长跑通过 · {a.usage_report()}")


if __name__ == "__main__":
    print(f"工具：{len(TOOLS)} 个\n")
    print("【1】100 Agent 并发")
    asyncio.run(ultra_concurrency())
    print("\n【2】双故障降级")
    asyncio.run(double_fault())
    print("\n【3】编码地狱")
    encoding_hell()
    print("\n【4】超长工具返回")
    huge_return()
    print("\n【5】Fuzz 1000 次")
    fuzz_1000()
    print("\n【6】100 轮长跑")
    asyncio.run(long_haul())
    print("\n=== M1 超级压测完成 ===")
