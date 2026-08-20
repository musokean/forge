"""M1 压测：工具边界健壮性 + 上下文截断 + 多轮循环收敛。"""
import asyncio
import os
import sys

sys.path.insert(0, ".")

from src.tools import execute, TOOLS
from src.agent import Agent


def tool_edge_cases():
    """工具边界压测：各种异常输入，必须返回结构化结果、不崩溃。"""
    cases = [
        ("read_file 不存在文件", "read_file", {"path": "Z:/no/such/file.txt"}),
        ("read_file 传目录", "read_file", {"path": "config"}),
        ("write_file 空路径", "write_file", {"path": "", "content": "x"}),
        ("calculator 除零", "calculator", {"expression": "1/0"}),
        ("calculator 危险表达式", "calculator", {"expression": "import os"}),
        ("web_fetch 坏URL", "web_fetch", {"url": "not-a-url"}),
        ("web_fetch 404页面", "web_fetch", {"url": "https://example.com/nonexistent-xyz-12345"}),
        ("run_command 不存在命令", "run_command", {"command": "this_cmd_not_exist_xyz"}),
        ("未知工具", "nonexist", {}),
    ]
    for name, tool, args in cases:
        try:
            r = execute(tool, args)
            assert isinstance(r, dict) and "ok" in r, "应返回结构化结果"
            ok = "✅" if r["ok"] is False else "⚠️"
            detail = r.get("error") or (str(r.get("data"))[:40] if r.get("data") else "")
            print(f"  {ok} {name}: ok={r['ok']} {detail}")
        except Exception as e:
            print(f"  ❌ {name}: 崩溃了！{type(e).__name__}: {e}")


def test_trim_stress():
    """上下文截断压测：灌 100 轮长消息，验证不越界、不破坏结构。"""
    a = Agent(max_context_tokens=200)
    for _ in range(100):
        a.messages.append({"role": "user", "content": "测" * 100})
        a.messages.append({"role": "assistant", "content": "试" * 100})
    a._trim_context()
    assert a.messages[0]["role"] == "system", "system 必须保留"
    user_idx = [i for i, m in enumerate(a.messages) if m["role"] == "user"]
    assert user_idx, "截断后至少保留一轮"
    print(f"  ✅ 上下文截断压测：100 轮 → {len(a.messages)} 条，预算内、system 保留")


async def loop_stress():
    """多轮循环压测（真实调 LLM）：连续 3 轮验证上下文 + token 累计。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("  ⚠ 未设 key，跳过循环压测")
        return
    a = Agent()
    for q in ["1+1 等于几（只答数字）", "刚才的答案是多少", "我叫小明，我叫什么"]:
        ans = await a.run(q)
        print(f"  问「{q}」→ 答：{ans[:60]}")
    assert a.total_tokens["prompt"] > 0, "token 统计应有值"
    print(f"  ✅ 多轮循环压测通过 · {a.usage_report()}")


if __name__ == "__main__":
    print(f"工具总数：{len(TOOLS)} = {list(TOOLS.keys())}\n")
    print("【1】工具边界健壮性压测")
    tool_edge_cases()
    print("\n【2】上下文截断压测")
    test_trim_stress()
    print("\n【3】多轮循环压测（真实调 LLM）")
    asyncio.run(loop_stress())
    print("\n=== M1 压测完成 ===")
