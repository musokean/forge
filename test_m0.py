"""M0 验收测试：跑真实 DeepSeek 调用，验证核心循环（对应测试计划 M0 用例）。"""
import asyncio
from src.agent import Agent


async def main():
    agent = Agent()

    print("=== 用例 1：纯问答（应直接回答，不调工具）===")
    r1 = await agent.run("1+1 等于几？")
    print("回答：", r1)

    print("\n=== 用例 2：工具调用（应调 calculator）===")
    r2 = await agent.run("帮我算一下 (3+5)*2 等于多少")
    print("回答：", r2)

    print("\n=== 用例 3：多轮上下文（先说我叫老大，再问名字）===")
    agent3 = Agent()
    await agent3.run("我叫老大")
    r3 = await agent3.run("我叫什么？")
    print("回答：", r3)


if __name__ == "__main__":
    asyncio.run(main())
