"""模型调用：openai SDK 网关 + 指数退避重试 + 模型降级（对应 A04b 框架做 I/O、A07 错误处理）。"""
import asyncio
import os

import httpx
import openai
from openai import AsyncOpenAI

from .config import resolve_model

_clients = {}  # 按 (base_url, api_key) 缓存客户端

# 全局调用超时（老大 2026-08-20：路由判断卡死根因——端点不可达时 openai 默认超时 600s × 3 次重试）
# ⚠ 必须用 httpx.Timeout 对象，不能传 dict：openai SDK ≥2.x 的 timeout 参数只接受 float/Timeout，
#   传 dict 会 TypeError（unsupported operand type(s) for +: 'float' and 'dict'）→ 被包装成 APIConnectionError
_CLIENT_TIMEOUT = httpx.Timeout(
    connect=8,    # 建立连接超时（端点不可达时快速失败，不再傻等）
    read=60,      # 读取响应超时（模型正常出字不会超）
    write=30,
    pool=30,
)

# 可重试的异常：限流 / 网络 / 超时 / 服务端 5xx
_RETRIABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _get_client(base_url: str, api_key: str):
    # 缓存 key 含事件循环 id：httpx 连接池绑定事件循环，跨 asyncio.run 复用会崩
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    key = (base_url, api_key, loop_id)
    if key not in _clients:
        # 全局超时（老大 2026-08-20：端点不可达时默认 600s 超时+3 次重试=卡死路由判断）：
        #   connect 8s 建立连接 · read 60s 等待响应 · write 30s · pool 30s
        _clients[key] = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=_CLIENT_TIMEOUT)
    return _clients[key]


async def _call_role(cfg, messages, role, tools=None, retries=3, base_delay=1.0):
    """单角色调用，带指数退避重试（A07：1s → 2s → 4s...）。"""
    m = resolve_model(cfg, role)
    api_key = m.get("api_key")
    if not api_key:
        raise ValueError(
            f"模型「{m.get('name') or role}」缺少可用的 API Key。"
            f"请在 config/models.yaml 的该模型下写入字面 api_key，或配置 api_key_env 并先导出对应环境变量。"
        )
    client = _get_client(m["base_url"], api_key)
    kwargs = {"model": m["model"], "messages": messages}
    if tools:
        kwargs["tools"] = tools

    last_err = None
    for attempt in range(retries):
        try:
            return await client.chat.completions.create(**kwargs)
        except _RETRIABLE as e:
            last_err = e
            if attempt == retries - 1:
                break
            await asyncio.sleep(base_delay * (2 ** attempt))  # 指数退避
    raise last_err


def _fallback_role(cfg, role):
    """取降级角色名（兼容 roles 两种写法：字符串别名 / {model, label, purpose} dict）。

    配置人性化改造后 roles 值多为 dict（如 `fallback: { model: qwen3_local_b, ... }`），
    直接当角色名传给 resolve_model 会 TypeError（dict 不可哈希）——地狱压测揪出的 bug #9。
    """
    fb = cfg.get("roles", {}).get("fallback")
    if isinstance(fb, dict):
        fb = fb.get("model")  # dict 写法：取 model 字段当角色/别名
    if not fb or fb == role:
        return None
    return fb


async def chat(cfg, messages, role="default", tools=None, retries=3):
    """先试主角色，失败降级到 fallback 角色；故障角色触发熔断后直接跳过（A07 降级 + #5 熔断）。"""
    from .circuit import get_circuit_registry, CircuitOpenError

    roles_to_try = [role]
    fb = _fallback_role(cfg, role)
    if fb:
        roles_to_try.append(fb)

    reg = get_circuit_registry(cfg)
    last_err = None
    for r in roles_to_try:
        br = reg.get(r)
        if not br.allow():
            # 熔断中：直接跳过该角色，不浪费重试退避，立刻走下一个角色
            last_err = CircuitOpenError(
                f"角色「{r}」熔断中（冷却剩余 {br.cooldown_remaining:.0f}s），已跳过"
            )
            continue
        try:
            resp = await _call_role(cfg, messages, r, tools=tools, retries=retries)
            br.record_success()
            return resp
        except Exception as e:
            br.record_failure()
            last_err = e
            continue  # 任何失败都降级到下一个角色（含 401/402/403 等账户级错误）
    raise last_err


async def stream_chat(cfg, messages, role="default", tools=None):
    """流式调用，逐块 yield (delta, usage)；失败降级到 fallback（A07）；故障角色熔断后跳过（#5）。

    delta 为消息增量（含 reasoning_content / content / tool_calls），usage 为 token 统计（流式末块返回）。
    """
    from .circuit import get_circuit_registry, CircuitOpenError

    roles_to_try = [role]
    fb = _fallback_role(cfg, role)
    if fb:
        roles_to_try.append(fb)

    reg = get_circuit_registry(cfg)
    last_err = None
    for r in roles_to_try:
        br = reg.get(r)
        if not br.allow():
            last_err = CircuitOpenError(
                f"角色「{r}」熔断中（冷却剩余 {br.cooldown_remaining:.0f}s），已跳过"
            )
            continue
        m = resolve_model(cfg, r)
        api_key = m.get("api_key")
        if not api_key:
            raise ValueError(f"模型「{m.get('name') or r}」缺少可用的 API Key（请在 models.yaml 配置 api_key 或 api_key_env）")
        client = _get_client(m["base_url"], api_key)
        base_kwargs = {"model": m["model"], "messages": messages}
        if tools:
            base_kwargs["tools"] = tools
        try:
            stream = await client.chat.completions.create(**base_kwargs, stream=True, stream_options={"include_usage": True})
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                delta = chunk.choices[0].delta if chunk.choices else None
                yield delta, usage
            br.record_success()
            return
        except Exception as e:
            last_err = e
            # 任何失败（含 401/402/403）都先降级：流式偶发中断 → 非流式重拿完整结果兜底
            try:
                resp = await client.chat.completions.create(**base_kwargs)
                yield resp.choices[0].message, getattr(resp, "usage", None)
                br.record_success()
                return
            except Exception:
                br.record_failure()
                continue
    raise last_err
