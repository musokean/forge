"""M1 验收测试：工具只读分级 / 上下文截断 / 模型降级配置 / token 统计。"""
import asyncio
import os
import sys

sys.path.insert(0, ".")

from src.tools import is_write, TOOLS
from src.agent import Agent, _estimate_tokens
from src.config import load_config


def test_readonly():
    assert is_write("read_file") is False
    assert is_write("calculator") is False
    assert is_write("write_file") is True
    assert TOOLS["write_file"]["read_only"] is False
    print("✅ 工具只读分级：write_file=写操作，read_file/calculator=只读")


def test_trim_context():
    a = Agent(max_context_tokens=100)
    for _ in range(20):
        a.messages.append({"role": "user", "content": "x" * 50})
        a.messages.append({"role": "assistant", "content": "y" * 50})
    before = len(a.messages)
    a._trim_context()
    after = len(a.messages)
    assert after < before, "截断后消息应变少"
    assert a.messages[0]["role"] == "system", "system 提示必须保留"
    assert a.messages[1]["role"] == "user", "截断后应从 user 轮开始"
    print(f"✅ 上下文截断：{before} 条 → {after} 条，system 保留、从 user 轮开始")


def test_fallback_config():
    cfg = load_config()
    assert cfg["roles"].get("fallback"), "roles 里应配 fallback 备用模型"
    print(f"✅ 模型降级配置：fallback = {cfg['roles']['fallback']}")


async def test_token_usage():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("⚠ 未设 DEEPSEEK_API_KEY，跳过 token 统计真实调用")
        return
    a = Agent()
    await a.run("1+1 等于几？请只回答数字")
    assert a.total_tokens["prompt"] > 0, "应有 prompt token 统计"
    print(f"✅ token 统计：{a.usage_report()}")


if __name__ == "__main__":
    test_readonly()
    test_trim_context()
    test_fallback_config()
    asyncio.run(test_token_usage())
    print("\n=== M1 测试完成 ===")
