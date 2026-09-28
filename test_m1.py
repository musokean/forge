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
    """降级链（A07）——断机制，不断本机配置。

    `config/models.yaml` 不入库（含 key）、每人不同，且默认配置已精简为单角色：
    「必须配 fallback」不是项目要求，只是可选增强（不配时 LLM 层失败直接抛 / 友好提示）。
    所以这里验证的是降级链解析机制本身：
      ① 至少要有 default 主角色；
      ② 配了 fallback（字符串别名 / dict 两种写法）→ 必须能解析出来；
      ③ 没配 → 降级链退化为「仅主角色」，合法。
    """
    from src.llm import _fallback_role

    cfg = load_config()
    roles = cfg.get("roles", {}) or {}
    assert roles.get("default"), "roles 里至少要有 default 主角色"

    fb = _fallback_role(cfg, "default")
    chain = ["default"] + ([fb] if fb else [])
    assert chain[0] == "default", "降级链首位必须是主角色"

    if roles.get("fallback"):
        assert fb, "配置了 fallback 但降级链没取到（_fallback_role 解析失败）"
        print(f"✅ 模型降级：fallback = {fb}，降级链 = {chain}")
    else:
        assert fb is None, "没配 fallback 却解析出了降级角色"
        print(f"✅ 模型降级：未配 fallback（精简配置合法），降级链 = {chain}")

    # 合成配置：两种写法都要能解析（dict 写法是地狱压测 bug #9，防回归）
    assert _fallback_role({"roles": {"default": "m1", "fallback": "m2"}}, "default") == "m2"
    assert _fallback_role({"roles": {"default": "m1", "fallback": {"model": "m3"}}}, "default") == "m3"
    assert _fallback_role({"roles": {"default": "m1", "fallback": "default"}}, "default") is None
    print("✅ 降级链解析：字符串别名 / dict 写法 / 自身不降级 三种形态正确")


async def test_token_usage():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("⚠ 未设 DEEPSEEK_API_KEY，跳过 token 统计真实调用")
        return
    a = Agent()
    await a.run("1+1 等于几？请只回答数字")
    assert a.total_tokens["prompt"] > 0, "应有 prompt token 统计"
    print(f"✅ token 统计：{a.usage_report()}")




# async + 需要真实模型/网络：pytest 不收集（否则报「async def 不被原生支持」误判失败）；
# 需要时直接跑 `python test_m1.py`（__main__ 里 asyncio.run 仍会执行）。
test_token_usage.__test__ = False

if __name__ == "__main__":
    test_readonly()
    test_trim_context()
    test_fallback_config()
    asyncio.run(test_token_usage())
    print("\n=== M1 测试完成 ===")
