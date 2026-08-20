"""核心 ReAct 循环 + 工程底盘：上下文截断、token 统计（对应 A01 架构、A05 上下文、A09 可观测）。"""
import asyncio
import json
import time

from .config import load_config, resolve_model
from .llm import chat, stream_chat
from .tools import get_tools_schema, execute, is_write
from .console import C, paint
from .structured import ask_structured, StructuredError
from .trace import Tracer
from .approval import Approver
from .skills import compose_prompt, schema_filter
from .spinner import spinner_start, spinner_stop
from .keypress import poll_key, read_guide_line


def _estimate_tokens(text):
    """粗略估算 token 数：中文≈1字1token，英文≈4字符1token；统一 len//2 保守估算。"""
    return max(1, len(text or "") // 2)


class Agent:
    def __init__(self, config_path="config/models.yaml", max_steps=10, max_context_tokens=8000,
                 role="default", system_prompt=None, name="forge", stream=True,
                 max_tool_output=1500, summary_ratio=0.85, approver=None, show_spinner=True):
        self._config_path = config_path  # 记住来源，支持热重载（/config 改完立刻生效）
        self.cfg = load_config(config_path)
        self.max_steps = max_steps  # 防死循环硬闸（A07）
        self.max_context_tokens = max_context_tokens  # 上下文 token 预算（A05）
        self.max_tool_output = max_tool_output  # 工具输出入历史前的最大字符数（A05b 杠杆②）
        self.summary_ratio = summary_ratio  # 触发滚动摘要的预算比例（A05b 杠杆②）
        self.approver = approver or Approver()  # #4 写操作审批层（默认 CLI 交互）
        self.role = role  # 绑定的模型角色（A17 多模型路由）
        # 系统提示：主对话（未显式传 system_prompt）叠加激活技能片段；辩论/并行等专用人设不叠加
        base_prompt = system_prompt or self._system_prompt()
        self._explicit_prompt = system_prompt is not None
        self._system = compose_prompt(base_prompt) if not self._explicit_prompt else base_prompt
        self.name = name  # 流式输出时的显示名前缀
        self.stream = stream  # 是否流式输出（并行子任务用 False 避免输出交错）
        self.show_spinner = show_spinner  # 等待模型响应时显示旋转动画（并行/辩论子任务关，防输出交错）
        # 对话历史：只初始化 system 提示，连续 run 保留上下文（多轮对话）
        self.messages = [{"role": "system", "content": self._system}]
        self.total_tokens = {"prompt": 0, "completion": 0}  # token 统计（A09）
        self._interrupts = 0  # 流式生成被打断次数（连续打断防死循环）
        # #7 trace：每角色/每步可观测（A09 补全）
        m = resolve_model(self.cfg, role)
        self.tracer = Tracer(agent_name=name, role=role, model=m.get("model"))

    def trace_report(self):
        """本 Agent 的 trace 流水（CLI /trace 用）。"""
        return self.tracer.render()

    def _system_prompt(self):
        return (
            "你是 forge（匠），一个多模型驱动的 AI 助手，定位是「把想法锻造成现实」。"
            "你能调用工具完成任务：文件操作 read_file/write_file/edit_file/list_files/search_file，"
            "联网 web_search/web_fetch，算数 calculator，执行命令 run_command，查时间 get_time，"
            "本地知识库 kb_search（检索知识库）/kb_add（把知识直接记入知识库）/kb_ingest（把本地文件或目录加入索引）。"
            "用户说「记到知识库/记住这个/沉淀要点」时用 kb_add；查本地资料/历史文档用 kb_search；"
            "需要查最新信息时优先用 web_search；"
            "搜索类问题搜 1-2 次后就应基于结果综合回答，不要反复搜索同一内容；"
            "拿到工具结果后，用简洁、准确的中文给出最终答案。"
        )

    def reload_config(self):
        """热重载模型/角色配置（在 CLI 里 /config 改完后调用）。

        只换模型配置，保留当前对话历史与 token 统计——改完模型能接着聊。
        """
        self.cfg = load_config(self._config_path)
        return self.cfg

    def refresh_system(self):
        """技能切换（/skill on|off）后重建系统提示：保留对话历史，只替换 system 首条。"""
        if self._explicit_prompt:
            return  # 辩论/并行专用人设不叠加技能
        self._system = compose_prompt(self._system_prompt())
        self.messages[0] = {"role": "system", "content": self._system}

    def reset(self):
        """新开一个对话（清空历史 + token 统计，只留 system 提示）。"""
        self.messages = [{"role": "system", "content": self._system}]
        self.total_tokens = {"prompt": 0, "completion": 0}

    # ---------- A05 上下文截断 ----------

    def _context_tokens(self):
        return sum(_estimate_tokens(m.get("content") or "") for m in self.messages)

    def _trim_context(self):
        """超 token 预算时，丢弃最旧的整轮（摘要占位消息保留，是已压缩信息）。"""
        while self._context_tokens() > self.max_context_tokens:
            user_idx = [i for i, m in enumerate(self.messages)
                        if m["role"] == "user" and not (m.get("content") or "").startswith("[早期对话摘要")]
            if len(user_idx) < 2:
                return  # 只剩一轮（或全是摘要），无法再截，保底
            del self.messages[user_idx[0]:user_idx[1]]  # 丢掉最早一轮

    # ---------- A05b 滚动摘要 + 工具输出裁剪（#3 上下文管理补全） ----------

    async def _summarize(self, text: str, purpose: str, max_in: int = 8000, max_out: int = 180):
        """用 fallback 便宜模型压缩文本（A05b 杠杆②）。失败返回 None，绝不阻塞主流程。"""
        if not text.strip():
            return ""
        from .llm import _fallback_role  # 延迟导入避免循环
        fb = _fallback_role(self.cfg, self.role)
        if fb is None:
            return None  # 没有降级角色可做摘要，交调用方兜底
        msgs = [
            {"role": "system", "content": "你是摘要助手，把输入压缩成要点，保留数字、路径、结论等关键信息，只输出摘要本身。"},
            {"role": "user", "content": f"[{purpose}]\n{text[:max_in]}"},
        ]
        try:
            resp = await chat(self.cfg, msgs, role=fb)
            self._add_usage(getattr(resp, "usage", None))  # 摘要调用诚实记账
            out = (resp.choices[0].message.content or "").strip()
            return out[:max_out] or None
        except Exception:
            return None  # 摘要失败不影响主流程

    async def _maybe_roll_summary(self):
        """预算超 85% 时，把最早的整轮历史压成摘要替代原文（比硬截断保留信息）。"""
        while self._context_tokens() > self.max_context_tokens * self.summary_ratio:
            # 排除摘要占位消息，防止把摘要当「用户轮」反复压缩（死循环根源）
            user_idx = [i for i, m in enumerate(self.messages)
                        if m["role"] == "user" and not (m.get("content") or "").startswith("[早期对话摘要")]
            if len(user_idx) < 2:
                self._trim_context()  # 只剩一轮：退回硬截断保底
                return  # 必须 return，否则 while 恒真死循环
            s, e = user_idx[0], user_idx[1]  # 最早的非摘要轮（含其 assistant/tool 消息）
            chunk = self.messages[s:e]
            del self.messages[s:e]
            flat = "\n".join(
                f"{m['role']}: {m.get('content') or ''}"
                + (f"  [工具 {m['tool_call_id']}]" if m.get("tool_call_id") else "")
                for m in chunk
            )
            summary = await self._summarize(flat, "把这段早期对话压缩成背景摘要", max_out=120)
            if summary:
                # 用 user/assistant 对存摘要（不破坏交替结构，兼容所有 OpenAI 兼容端点）
                self.messages.insert(s, {"role": "user", "content": "[早期对话摘要，作为背景参考，不需要回应]"})
                self.messages.insert(s + 1, {"role": "assistant", "content": summary})
            # 摘要失败则视为已压缩（等价原硬截断，不阻塞）

    async def _clip_tool_output(self, text: str) -> str:
        """大段工具输出入历史前压缩（A05b 杠杆②）：超阈值走摘要，失败截断保底。"""
        if len(text) <= self.max_tool_output:
            return text
        summary = await self._summarize(text, f"压缩工具返回内容（原 {len(text)} 字符），保留关键数据", max_in=8000, max_out=300)
        if summary:
            return f"[工具输出已压缩 {len(text)} 字符 → 摘要]\n{summary}"
        return text[:self.max_tool_output] + f"\n…[已截断，原 {len(text)} 字符]"


    # ---------- A09 token 统计 ----------

    def _add_usage(self, usage):
        """累计 token（流式末块传入 usage 对象）。"""
        if usage:
            self.total_tokens["prompt"] += usage.prompt_tokens or 0
            self.total_tokens["completion"] += usage.completion_tokens or 0

    async def run_structured(self, task: str, schema):
        """结构化输出版 run：要求模型返回符合 schema 的 JSON 对象（对应 #6 / A27）。

        解析失败自动重试；成功返回校验后的 Pydantic 对象（彻底失败返回 None）。
        非流式、记 token 并写入历史（存 JSON 字符串，便于后续引用）。
        """
        self.messages.append({"role": "user", "content": task})
        self._trim_context()
        self._t_start = time.monotonic()
        self.tracer.record("run_start", task=task)
        try:
            obj, resp = await ask_structured(self.cfg, self.messages, schema, role=self.role)
        except StructuredError:
            self._finish_trace()
            return None  # 兜底：保留用户问题于历史，交给调用方决定
        self._add_usage(getattr(resp, "usage", None))
        self.messages.append({
            "role": "assistant",
            "content": json.dumps(obj.model_dump(), ensure_ascii=False),
        })
        self._finish_trace(json.dumps(obj.model_dump(), ensure_ascii=False))
        return obj

    def usage_report(self):
        p, c = self.total_tokens["prompt"], self.total_tokens["completion"]
        return f"本次累计 tokens：prompt {p} · completion {c} · 共 {p + c}"

    def _print_status(self):
        """本次回复的状态栏：角色 + 耗时 + token 用量 + 上下文容量（A09 可观测，实时显示）。"""
        if not self.stream:
            return  # 非流式（并行子任务）不打印，避免输出交错
        total = time.monotonic() - self._t_start
        p, c = self.total_tokens["prompt"], self.total_tokens["completion"]
        dp = (p - self._tok_before["prompt"]) + (c - self._tok_before["completion"])
        ctx = self._context_tokens()
        pct = min(100.0, ctx / max(1, self.max_context_tokens) * 100)
        wait = f"{self._ttft:.1f}s" if self._ttft is not None else "—"
        print()
        print(paint(f"  [{self.name}] ", C.LIGHT_BLUE + C.BOLD)
              + paint(f"⏱ 首字 {wait} · 总 {total:.1f}s"
                      f" ｜ 📊 本次 +{dp} tok · 累计 {p + c}"
                      f" ｜ 🧠 上下文 {pct:.0f}% ({ctx}/{self.max_context_tokens})", C.SKY_DIM))

    # ---------- #7 trace 打点 ----------

    def _trace_llm(self, step, t0):
        """模型调用完成：记录步进、本次增量 token、耗时。"""
        tok = ((self.total_tokens["prompt"] - self._tok_before["prompt"])
               + (self.total_tokens["completion"] - self._tok_before["completion"]))
        self.tracer.record("llm_step", step=step, tokens=max(0, tok), ms=(time.monotonic() - t0) * 1000)

    def _trace_tool(self, name, args, result_text, t0):
        """工具调用完成：记录名称、参数摘要、结果摘要、耗时。"""
        self.tracer.record("tool", name=name, arg_summary=str(args)[:200],
                           result_summary=result_text[:150], ms=(time.monotonic() - t0) * 1000)

    def _execute_with_approval(self, name, args):
        """执行工具；写操作先过审批层（#4）。拒绝时返回拒绝结果喂回模型调整方案。"""
        if is_write(name) and not self.approver.approve(name, str(args)[:200]):
            return {"ok": False, "error": f"用户拒绝了该写操作（{name}），请调整方案或改用只读方式"}
        return execute(name, args)

    def _finish_trace(self, content=None):
        """run() 出口统一收尾：answer（若有）+ run_end（总耗时、总 token）。"""
        if content is not None:
            self.tracer.record("answer", content=content)
        p, c = self.total_tokens["prompt"], self.total_tokens["completion"]
        self.tracer.record("run_end", ms=(time.monotonic() - self._t_start) * 1000,
                           prompt_tokens=p, completion_tokens=c)

    # ---------- 反思自纠错（A07 组合拳末环） ----------

    async def _maybe_reflect(self, task: str, answer: str) -> str:
        """配置启用时：评审一次，低分带意见重答（最多 max_rounds 轮）。失败静默保留原答案。"""
        r = (self.cfg.get("reflect") or {})
        if not r.get("enabled", False):
            return answer
        max_rounds = int(r.get("max_rounds", 1))
        if max_rounds <= 0 or not answer.strip():
            return answer
        judge = r.get("judge_role", "fallback")
        min_score = float(r.get("min_score", 6))
        for _ in range(max_rounds):
            from .reflect import evaluate_answer, refine_answer
            verdict = await evaluate_answer(self.cfg, task, answer, judge_role=judge)
            if verdict is None or verdict["score"] >= min_score:
                return answer  # 评审通过或评审失败：接受现状
            issues = "；".join(verdict["issues"]) if verdict["issues"] else "答案质量不足"
            feedback = f"{issues}。{verdict['suggestion']}".strip("。")
            print(paint(f"  🔧 反思评审 {verdict['score']}/10：{feedback}", C.SKY))
            if self.stream:
                print()  # 结束已打印的答案行
            new = await refine_answer(self.cfg, task, answer, feedback, role=self.role)
            if new == answer:
                return answer
            answer = new
            if self.stream:
                print(paint(f"  🔧 修正版：", C.SKY), end="")
                print(answer, end="", flush=True)
                print()
        return answer

    # ---------- 核心循环 ----------

    async def run(self, task: str) -> str:
        self.messages.append({"role": "user", "content": task})
        self._interrupts = 0  # 每次 run 重置打断计数
        await self._maybe_roll_summary()  # 预算高时先滚动摘要（A05b），兜底硬截断
        # —— 可观测：本次回复的计时与 token 快照（A09）——
        self._tok_before = dict(self.total_tokens)
        self._t_start = time.monotonic()
        self._ttft = None  # 首字等待（首个响应块到达）
        self.tracer.record("run_start", task=task)
        last_sig = None   # 上一轮工具调用签名
        repeat = 0        # 连续重复次数
        force_answer = False  # 强制收敛标志（A01/A07 防死循环）
        for step in range(self.max_steps):
            if step >= self.max_steps - 1:
                force_answer = True
            tools = None if force_answer else schema_filter(get_tools_schema())

            if not self.stream:
                # 非流式（并行子任务用）：不打印，返回完整结果
                t0 = time.monotonic()
                spin = spinner_start("💭 生成中") if self.show_spinner else None
                try:
                    resp = await chat(self.cfg, self.messages, role=self.role, tools=tools)
                except Exception as e:
                    spinner_stop(spin)
                    self._finish_trace()
                    ep = resolve_model(self.cfg, self.role).get("base_url", "?")
                    return f"⚠ 模型调用失败（端点 {ep}）：{e}（已重试并尝试降级通道，请检查端点/余额/网络）"
                spinner_stop(spin)
                self._add_usage(resp.usage)
                self._trace_llm(step, t0)
                msg = resp.choices[0].message
                reasoning = getattr(msg, "reasoning_content", None) or ""
                if msg.tool_calls and not force_answer:
                    sig = tuple((tc.function.name, tc.function.arguments) for tc in msg.tool_calls)
                    repeat = repeat + 1 if sig == last_sig else 1
                    last_sig = sig
                    if repeat >= 3:
                        self.messages.append({"role": "user", "content": "你已连续多次重复调用相同工具、未获得新信息。请停止调用工具，基于已有信息直接给出最终答案。"})
                        force_answer = True
                        continue
                    assistant_msg = {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                            for tc in msg.tool_calls
                        ],
                    }
                    if reasoning:
                        assistant_msg["reasoning_content"] = reasoning
                    self.messages.append(assistant_msg)
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments or "{}")
                        t1 = time.monotonic()
                        result = self._execute_with_approval(tc.function.name, args)
                        text = json.dumps(result, ensure_ascii=False)
                        self._trace_tool(tc.function.name, args, text, t1)
                        self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                              "content": await self._clip_tool_output(text)})
                else:
                    answer = msg.content or ""
                    answer = await self._maybe_reflect(task, answer)
                    self._finish_trace(answer)
                    return answer
                continue

            reasoning_parts = []
            content_parts = []
            tool_calls_acc = {}   # index -> {"id", "name", "arguments"}
            reasoning_started = False
            content_started = False
            interrupted = False   # 用户打断标志
            guide_text = None     # 引导输入（非 None 时打断并重新生成）

            # 流式接收：思考过程与答案边生成边打印（打字机效果）
            # 手动迭代（__anext__）而非 async for：等待首字阶段每 0.5s 轮询键盘，
            # 让「连接端点/生成首个 token」期间也能 Esc 打断 / 任意键引导（老大 2026-08-20 反馈）
            t0 = time.monotonic()
            spin = spinner_start("💭 思考中") if self.show_spinner else None
            stream = None
            try:
                stream = stream_chat(self.cfg, self.messages, role=self.role, tools=tools)
                it = stream.__aiter__()

                def _check_key():
                    """打断检测：Esc 中断 / 任意键进入引导。返回 True 表示已打断。"""
                    nonlocal interrupted, guide_text, spin
                    if not (self.show_spinner and not self._explicit_prompt):
                        return False
                    k = poll_key()
                    if k == "ESC":
                        interrupted = True
                        spinner_stop(spin)
                        spin = None
                        return True
                    if k is not None:
                        interrupted = True
                        guide_text = read_guide_line()
                        spinner_stop(spin)
                        spin = None
                        return True
                    return False

                while True:
                    # 首字阶段：0.5s 超时轮询，让等待期可打断
                    if self._ttft is None:
                        try:
                            delta, usage = await asyncio.wait_for(it.__anext__(), timeout=0.5)
                        except asyncio.TimeoutError:
                            if _check_key():
                                break
                            continue
                    else:
                        try:
                            delta, usage = await it.__anext__()
                        except StopAsyncIteration:
                            break
                    # 后续 chunk 间也轮询（边生成边可打断）
                    if _check_key():
                        break
                    if self._ttft is None:
                        self._ttft = time.monotonic() - self._t_start  # 首字等待
                        spinner_stop(spin)  # 首个响应块到达：停动画
                        spin = None
                    if usage:
                        self._add_usage(usage)
                    if not delta:
                        continue
                    r = getattr(delta, "reasoning_content", None)
                    if r:
                        reasoning_parts.append(r)
                        if not reasoning_started:
                            print(paint("💭 ", C.SKY_DIM), end="", flush=True)
                            reasoning_started = True
                        print(paint(r, C.SKY_DIM), end="", flush=True)
                    c = getattr(delta, "content", None)
                    if c:
                        if not content_started:
                            if reasoning_parts:
                                print()  # 结束思考行
                            print(paint(self.name, C.LIGHT_BLUE + C.BOLD) + paint(" › ", C.SKY_DIM), end="", flush=True)
                            content_started = True
                        content_parts.append(c)
                        print(c, end="", flush=True)
                    for tc in getattr(delta, "tool_calls", None) or []:
                        acc = tool_calls_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] += tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments

            except Exception as e:
                # A07 兜底：流式链路最终失败（重试+降级全失败）→ 友好提示，不崩 REPL
                spinner_stop(spin)
                print()
                self._print_status()
                self._finish_trace()
                ep = resolve_model(self.cfg, self.role).get("base_url", "?")
                return f"⚠ 模型调用失败（端点 {ep}）：{e}（已重试并尝试降级通道，请检查端点/余额/网络，或 /config 切换模型）"
            finally:
                if stream is not None:
                    try:
                        await stream.aclose()  # 手动迭代需显式关闭，避免 httpx 连接挂起
                    except Exception:
                        pass

            reasoning = "".join(reasoning_parts)
            content = "".join(content_parts)
            spinner_stop(spin)  # 空流兜底（正常路径 spin 已在首字时停）
            self._trace_llm(step, t0)  # 流式模型调用完成打点

            # —— 打断处理：Esc 中断 → 返回已生成内容；带引导 → 塞回历史重新生成 ——
            if interrupted:
                print()  # 结束当前输出行
                if guide_text:
                    self._interrupts += 1
                    if self._interrupts >= 3:
                        print(paint("  ⏹ 已连续打断 3 次，中止本次生成。可换问题重新问。", C.SKY))
                        self._finish_trace()
                        return "（已连续打断，中止）"
                    # 已生成部分作为 assistant 消息入历史，引导作为新 user 消息 → 重新生成
                    self.messages.append({"role": "assistant", "content": content or "（未完成，用户打断）"})
                    self.messages.append({"role": "user",
                                          "content": f"【用户打断并给了新指示】请忽略刚才的回答方向，按以下引导重新回答：{guide_text}"})
                    print(paint(f"  ✍ 已收到引导，重新生成中…（{guide_text[:40]}）", C.SKY))
                    continue  # 进入下一轮 ReAct 循环，模型看到引导后重新生成
                # 纯 Esc 中断：返回已生成内容
                self._print_status()
                if content:
                    print(paint("  ⏹ 已中断（Esc）", C.SKY))
                    self._finish_trace(content + "\n（已中断）")
                    return content + "\n（已中断）"
                self._finish_trace()
                print(paint("  ⏹ 已中断", C.SKY))
                return "（已中断）"

            if tool_calls_acc and not force_answer:
                if content or reasoning:
                    print()  # 结束上一行
                tool_calls = [
                    {"id": a["id"], "type": "function", "function": {"name": a["name"], "arguments": a["arguments"]}}
                    for _, a in sorted(tool_calls_acc.items())
                ]
                sig = tuple((tc["function"]["name"], tc["function"]["arguments"]) for tc in tool_calls)
                repeat = repeat + 1 if sig == last_sig else 1
                last_sig = sig
                if repeat >= 3:
                    self.messages.append({
                        "role": "user",
                        "content": "你已连续多次重复调用相同工具、未获得新信息。请停止调用工具，基于已有信息直接给出最终答案。",
                    })
                    force_answer = True
                    continue
                assistant_msg = {"role": "assistant", "content": content or None, "tool_calls": tool_calls}
                if reasoning:
                    assistant_msg["reasoning_content"] = reasoning  # 工具调用时必须保留 CoT
                self.messages.append(assistant_msg)
                names = [tc["function"]["name"] for tc in tool_calls]
                warn = " ⚠写操作" if any(is_write(n) for n in names) else ""
                print(paint(f"  ⚙ 调用工具：{', '.join(names)}{warn}", C.SKY))
                for tc in tool_calls:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    t1 = time.monotonic()
                    result = self._execute_with_approval(tc["function"]["name"], args)
                    text = json.dumps(result, ensure_ascii=False)
                    self._trace_tool(tc["function"]["name"], args, text, t1)
                    self.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                          "content": await self._clip_tool_output(text)})
            else:
                print()  # 结束答案行
                self._print_status()
                answer = content or ""
                answer = await self._maybe_reflect(task, answer)
                self._finish_trace(answer)
                return answer

        self._print_status()
        self._finish_trace()
        return "达到最大步数，任务未收敛"
