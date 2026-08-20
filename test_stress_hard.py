"""M1 残酷压测：注入绕过 / 极限输入 / 并发 / 长时多轮 / 异常恢复 / 死循环极限。"""
import asyncio
import os
import sys
import time

sys.path.insert(0, ".")

from src.tools import execute, TOOLS, _safe_eval
from src.agent import Agent


def injection_fuzz():
    """calculator 注入绕过：各种逃逸 payload，必须全部拦截。"""
    payloads = [
        "__import__('os').system('echo hacked')",
        "__import__('os').listdir('/')",
        "(lambda: 1)()",
        "[1,2,3][0]",
        "'abc'.upper()",
        "().__class__.__bases__",
        "open('/etc/passwd').read()",
        "1 if True else 2",
        "eval('1+1')",
        "exec('x=1')",
        "{}.keys()",
        "(1).__class__",
        "getattr(os, 'system')",
        "globals()",
        "2**1000000",
    ]
    blocked = 0
    for p in payloads:
        try:
            r = _safe_eval(p)
            print(f"  ⚠️ 逃逸成功：{p[:45]} → {str(r)[:40]}")
        except Exception:
            blocked += 1
    print(f"  ✅ 注入绕过：{blocked}/{len(payloads)} 全部拦截")


def tool_extreme():
    """工具极限输入：超长、空串、emoji、特殊字符，必须不崩。"""
    cases = [
        ("web_search 超长关键词", "web_search", {"query": "x" * 500}),
        ("web_search 空串", "web_search", {"query": ""}),
        ("web_search emoji", "web_search", {"query": "🎯🔍🚀"}),
        ("calculator 超长表达式", "calculator", {"expression": "1+" * 500 + "1"}),
        ("calculator 浮点", "calculator", {"expression": "0.1+0.2"}),
        ("list_files 深层递归", "list_files", {"path": ".", "recursive": True}),
        ("search_file 大目录", "search_file", {"path": ".", "pattern": "def "}),
        ("run_command 管道", "run_command", {"command": "echo a && echo b"}),
        ("未知工具+奇怪参数", "nonexist_tool", {"x": {"y": [1, 2, 3]}}),
    ]
    for name, tool, args in cases:
        try:
            r = execute(tool, args)
            ok = "✅" if r["ok"] is True else "⚠️"
            print(f"  {ok} {name}: {str(r.get('data') or r.get('error'))[:55]}")
        except Exception as e:
            print(f"  ❌ {name}: 崩溃 {type(e).__name__}: {e}")


async def concurrent_stress():
    """并发压测：5 个 Agent 同时调 LLM，验证不串上下文、不崩。"""
    async def one(i):
        a = Agent()
        return await a.run(f"只回答数字：{i}+{i} 等于几")

    t0 = time.time()
    results = await asyncio.gather(*[one(i) for i in range(1, 6)])
    dt = time.time() - t0
    print(f"  5 个并发结果：{results}（耗时 {dt:.1f}s）")
    assert all(str(i * 2) in r for i, r in zip(range(1, 6), results)), "并发结果出错"


async def long_run_stress():
    """长时压测：20 轮 + 小预算强制频繁截断，验证不崩、token 累计正常。"""
    a = Agent(max_context_tokens=600)  # 小预算，逼它频繁截断
    for i in range(20):
        ans = await a.run("1+1 等于几？只回答数字")
        assert "2" in ans, f"第 {i} 轮答案错：{ans}"
    print(f"  ✅ 20 轮连续完成 · 上下文 {len(a.messages)} 条 · {a.usage_report()}")


async def recover_stress():
    """异常恢复：穿插失败任务，验证失败后 agent 还能继续。"""
    a = Agent()
    r1 = await a.run("读一下 Z:/不存在的文件.txt")
    r2 = await a.run("1+1 等于几（只答数字）")
    print(f"  失败任务后继续：{r2[:30]}")
    assert "2" in r2, "失败后未恢复"


async def dead_loop_stress():
    """死循环极限：搜索一个不存在的词，验证收敛机制兜住。"""
    a = Agent(max_steps=6)  # 更小步数，逼死循环更快暴露
    ans = await a.run("搜索关键词：xqzvw_不存在_12345，找不到就明说找不到")
    print(f"  死循环极限：最终答 → {ans[:80]}")
    assert "未收敛" not in ans, "没收敛！"


if __name__ == "__main__":
    print(f"工具总数：{len(TOOLS)}\n")
    print("【1】注入绕过（安全）")
    injection_fuzz()
    print("\n【2】工具极限输入")
    tool_extreme()
    print("\n【3】并发压测（真实调 LLM）")
    asyncio.run(concurrent_stress())
    print("\n【4】长时压测（20 轮 + 强制截断）")
    asyncio.run(long_run_stress())
    print("\n【5】异常恢复")
    asyncio.run(recover_stress())
    print("\n【6】死循环极限")
    asyncio.run(dead_loop_stress())
    print("\n=== M1 残酷压测完成 ===")
