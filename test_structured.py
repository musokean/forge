"""#6 结构化输出测试：JSON 抽取鲁棒性 + Pydantic 校验 + 失败重试 + 并行/辩论结构化交接。

框架无关：用 unittest.mock 替换 LLM，pytest 或直接 `python test_structured.py` 都能跑。
"""
import asyncio
import sys
import unittest.mock as mock

sys.path.insert(0, ".")

from pydantic import BaseModel

import src.structured as S
from src.structured import RoleBrief, ask_structured, extract_json
from src.agent import Agent
from src.orchestrator import debate, run_parallel


# ---------- mock LLM ----------
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


def _coro(r):
    async def _f(*a, **k):
        return r
    return _f


def _fake_chat(responses):
    it = iter(responses)

    async def _c(cfg, messages, role="default"):
        return next(it)
    return _c


class Person(BaseModel):
    name: str
    age: int


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('好的，这是结果：\n{"x": 2}\n谢谢') == {"x": 2}
    assert extract_json('{"a": 1,}') == {"a": 1}          # 尾随逗号
    assert extract_json("{'a': 1}") == {"a": 1}           # 单引号
    print("  ✅ extract_json 鲁棒性（围栏/散文/尾逗号/单引号）")


def test_ask_structured_happy():
    resp = _Resp('{"name": "张三", "age": 30}')
    with mock.patch.object(S, "chat", _coro(resp)):
        obj, _ = asyncio.run(ask_structured({}, [{"role": "user", "content": "你是谁"}], Person))
    assert obj.name == "张三" and obj.age == 30
    print("  ✅ ask_structured 成功路径返回校验对象")


def test_ask_structured_retry():
    bad = _Resp("这不是 JSON，我忘了格式")
    good = _Resp('{"name": "李四", "age": 22}')
    with mock.patch.object(S, "chat", _fake_chat([bad, good])):
        obj, _ = asyncio.run(ask_structured({}, [], Person, retries=2))
    assert obj.name == "李四"  # 第一次失败后重试成功
    print("  ✅ ask_structured 首错后重试成功")


def test_ask_structured_exhausted():
    bad = _Resp("还是不对")
    with mock.patch.object(S, "chat", _fake_chat([bad, bad, bad])):
        try:
            asyncio.run(ask_structured({}, [], Person, retries=2))
            assert False, "应抛 StructuredError"
        except S.StructuredError:
            pass
    print("  ✅ ask_structured 重试耗尽抛 StructuredError")


def test_agent_run_structured():
    resp = _Resp('{"name": "王五", "age": 40}')
    with mock.patch.object(S, "chat", _coro(resp)):
        a = Agent(stream=False)
        obj = asyncio.run(a.run_structured("告诉我你是谁", Person))
    assert obj.age == 40
    assert a.total_tokens["prompt"] == 10 and a.total_tokens["completion"] == 5
    assert a.messages[-1]["role"] == "assistant"
    print("  ✅ Agent.run_structured 记账 + 历史写入")


def test_run_parallel_structured():
    r1 = _Resp('{"name":"t1","stance":"s1","key_points":["a"],"confidence":0.8,"source":"x"}')
    r2 = _Resp('{"name":"t2","stance":"s2","key_points":["b"],"confidence":0.6,"source":"y"}')
    with mock.patch.object(S, "chat", _fake_chat([r1, r2])):
        results = asyncio.run(run_parallel(["任务1", "任务2"], structured=True))
    assert all(isinstance(r, RoleBrief) for r in results)
    assert results[0].name == "t1" and results[1].confidence == 0.6
    print("  ✅ run_parallel(structured=True) 返回 RoleBrief 列表")


def test_debate_judge_structured():
    judge = _Resp('{"name":"裁判","stance":"结论","key_points":["p"],"confidence":0.9,"source":"讨论"}')
    with mock.patch.object(S, "chat", _coro(judge)):
        brief = asyncio.run(debate("某个问题", rounds=1, judge_structured=True))
    assert isinstance(brief, RoleBrief)
    assert brief.confidence == 0.9
    print("  ✅ debate(judge_structured=True) 裁判返回 RoleBrief")


if __name__ == "__main__":
    fns = [test_extract_json, test_ask_structured_happy, test_ask_structured_retry,
           test_ask_structured_exhausted, test_agent_run_structured,
           test_run_parallel_structured, test_debate_judge_structured]
    for fn in fns:
        fn()
    print("\n=== #6 结构化输出测试全过 ===")
