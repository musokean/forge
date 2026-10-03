"""#17 客户端执行器 · Computer Use 四角色循环（A25「防幻觉」核心）。

A25 的原话：**别让一个模型又当大脑又当手还当裁判**。所以这里四个角色是**四次独立调用**、
各有各的 system prompt、可以在配置里绑不同模型，而且**关键信息不对称**：

    Planner     把模糊任务拆成可执行步骤（只看任务 + 设备能力）
    Executor    针对当前步骤产出**一个**动作（cap/action/args/expect）——它是「手」
    Evaluator   只看**原始证据**（执行回传的 output/error/extra/截图），判定该步是否真的成功
                ——**看不到 Executor 的理由**，所以执行者的「我成功了」骗不过它
    Supervisor  连续失败达阈值时重新规划，或判定任务不可完成并中止（跳死循环）

配套机制（都对齐既有模块）：
  · 步数上限 / 连续失败阈值 / 重规划上限（对应 A07 防死循环）
  · 每步一条结构化日志（`cua_role`：角色 / 模型 / 耗时 / token）→ 四角色真的分开了可验证
  · 结论必须带 `[W 推断]` 语义：Evaluator 的判定只依据证据，没有证据就算不通过

配置（`config/models.yaml` 的 `cua` 段）：

    cua:
      enabled: true
      device_id: ""          # 默认用第一台在线执行器
      max_steps: 12
      max_failures: 2        # 连续失败到这个数 → 交给 Supervisor 重规划
      max_replans: 2
      screenshot_each_step: true
      roles:                 # 四角色绑模型（想更强就换成别的 alias）
        planner: default
        executor: default
        evaluator: default
        supervisor: default
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

from pydantic import BaseModel, Field

from .executor_hub import ExecutorError, get_hub
from .structured import StructuredError, ask_structured

ROLE_KEYS = ("planner", "executor", "evaluator", "supervisor")

DEFAULT_CUA = {
    "enabled": True,
    "device_id": "",
    "max_steps": 12,
    "max_failures": 2,
    "max_replans": 2,
    "screenshot_each_step": True,
    "vision": False,
    "evidence_clip": 2000,
}

# ── 四个角色的输出结构（各自一份，互不混用）───────────────
class PlanStep(BaseModel):
    goal: str = Field(description="这一步要达成什么（一句话）")
    hint: str = Field(default="", description="建议用哪个能力、注意什么")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description="按顺序的步骤列表")
    done_when: str = Field(default="", description="什么状态算任务完成")


class Action(BaseModel):
    cap: str = Field(description="shell / read_file / write_file / list_dir / screenshot / input / done")
    action: str = Field(default="", description="cap=input 时是 click/move/type/key/scroll；done 时留空")
    args: dict = Field(default_factory=dict, description="能力参数，如 {command: ...} 或 {x,y}")
    expect: str = Field(default="", description="做完后应该看到什么（给评估者对照）")
    why: str = Field(default="", description="为什么这么做（不参与评估，仅供人工追溯）")


class Verdict(BaseModel):
    ok: bool = Field(description="依据证据，这一步是否真的达成")
    reason: str = Field(default="", description="判断依据（引用证据里的具体内容）")
    next_hint: str = Field(default="", description="给下一步的提示（可选）")


class Replan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list, description="重新规划后的剩余步骤")
    give_up: bool = Field(default=False, description="判定无法完成时置 true")
    reason: str = Field(default="", description="为什么重规划/放弃")


class CuaError(Exception):
    pass


class CuaStep:
    def __init__(self, index, goal, cap="", action="", args=None, ok=False, reason="", evidence=None):
        self.index = index
        self.goal = goal
        self.cap = cap
        self.action = action
        self.args = dict(args or {})
        self.ok = ok
        self.reason = reason
        self.evidence = dict(evidence or {})

    def as_dict(self):
        return {"index": self.index, "goal": self.goal, "cap": self.cap, "action": self.action,
                "args": self.args, "ok": self.ok, "reason": self.reason,
                "evidence": {k: (str(v)[:400] if k != "output" else str(v)[:400])
                             for k, v in self.evidence.items()}}

    def __repr__(self):
        return f"<CuaStep #{self.index} {self.cap}:{self.action} ok={self.ok}>"


class CuaResult:
    def __init__(self, task, device_id, ok=False, reason="", steps=None, replans=0, ms=0.0, roles=None):
        self.task = task
        self.device_id = device_id
        self.ok = ok
        self.reason = reason
        self.steps = steps or []
        self.replans = replans
        self.ms = ms
        self.roles = roles or {}

    def as_dict(self):
        return {"ok": self.ok, "device_id": self.device_id, "reason": self.reason,
                "replans": self.replans, "ms": round(self.ms, 1),
                "steps": [s.as_dict() for s in self.steps], "roles": self.roles}


# ── 四个角色的 system prompt（分开写，互不复用）──────────────
SYS_PLANNER = ("你是 Computer Use 流水线里的**规划者 Planner**。"
               "把用户的模糊任务拆成 1-6 个**可验证**的步骤，每步都要能通过一次工具调用看到结果。"
               "你只能看到设备声明的能力，不要规划它做不到的动作（例如设备没有 screenshot 就别规划看图）。")
SYS_EXECUTOR = ("你是 Computer Use 流水线里的**执行者 Executor**。"
                "针对当前步骤，只给出**一个**动作（cap/action/args），并写清 expect（做完后应该看到什么）。"
                "**任务一旦达成（或证据已能证明达成），立刻返回 cap=done**，不要重做已完成的事、也不要去试探"
                "无关目录——每一步都有成本。文件类动作的路径必须落在「可写目录」之内（相对路径按该目录解析）。"
                "你**不负责判断成败**——那由评估者依据原始证据决定。")
SYS_EVALUATOR = ("你是 Computer Use 流水线里的**评估者 Evaluator**。只看下面给出的**原始证据**"
                 "（命令输出/错误码/文件内容/截图说明），判定这一步是否真的达成了目标。"
                 "证据不足、只看到「我说完成了」而没有实际结果时，一律判 ok=false。"
                 "不要相信任何未经证据支持的说法，也不要去猜执行者的意图。")
SYS_SUPERVISOR = ("你是 Computer Use 流水线里的**监督者 Supervisor**。"
                  "根据失败轨迹判断：是走偏了、参数错了、还是任务本身就做不到。"
                  "能给新思路就重新规划剩余步骤；确认做不到就把 give_up 置 true 并说明原因。")


class ComputerUseLoop:
    """四角色闭环：Planner → (Executor → 执行 → Evaluator) × N → Supervisor 兜底。"""

    def __init__(self, cfg=None, hub=None, roles=None, caller=None):
        conf = dict(DEFAULT_CUA)
        if isinstance(cfg, dict):
            conf.update({k: v for k, v in (cfg.get("cua") or {}).items() if v is not None})
        self.conf = conf
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.hub = hub if hub is not None else get_hub(cfg)
        self.roles = dict(conf.get("roles") or {})
        for k in ROLE_KEYS:
            self.roles.setdefault(k, "default")
        if roles:
            self.roles.update(roles)
        self._caller = caller          # 测试注入：async (role_key, schema, sys, user) -> obj
        self.role_calls = []           # 每次角色调用留痕（角色/模型/耗时）

    # ---------- 角色调用 ----------
    async def _call(self, role_key, schema, system, user, retries=1):
        alias = self.roles.get(role_key) or "default"
        t0 = time.time()
        if self._caller is not None:
            obj = await self._caller(role_key, schema, system, user)
            resp = None
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            try:
                obj, resp = await ask_structured(self.cfg, messages, schema, role=alias, retries=retries)
            except StructuredError as e:
                raise CuaError(f"角色 {role_key} 的模型输出不符合约定结构：{e}")
        ms = (time.time() - t0) * 1000
        usage = {}
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                         "completion_tokens": getattr(u, "completion_tokens", None)}
        except Exception:
            usage = {}
        entry = {"role": role_key, "model": alias, "ms": round(ms, 1), **usage}
        self.role_calls.append(entry)
        try:
            from .logging_setup import log_event

            log_event("INFO", "cua_role", **entry)
        except Exception:
            pass
        return obj

    # ---------- 各角色 ----------
    async def make_plan(self, task, device):
        dev = self._device_brief(device)
        user = (f"任务：{task}\n\n可用执行器：{dev}\n\n请给出步骤列表。")
        return await self._call("planner", Plan, SYS_PLANNER, user)

    async def choose_action(self, task, step, evidence, device):
        user = (f"总任务：{task}\n当前步骤：{step.goal}（提示：{step.hint or '无'}）\n"
                f"可用执行器：{self._device_brief(device)}\n"
                f"上一步的证据：{self._evidence_text(evidence)}\n\n请给出下一步动作。")
        return await self._call("executor", Action, SYS_EXECUTOR, user)

    async def judge(self, task, step_goal, action, evidence):
        user = (f"总任务：{task}\n这一步的目标：{step_goal}\n"
                f"执行的动作：cap={action.cap} action={action.action} args={json.dumps(action.args, ensure_ascii=False)}\n"
                f"执行者声称的预期：{action.expect or '（没写）'}\n"
                f"**原始证据**：{self._evidence_text(evidence)}\n\n"
                "只依据上面的原始证据判断这一步是否达成。")
        return await self._call("evaluator", Verdict, SYS_EVALUATOR, user)

    async def judge_completion(self, task, trace):
        """计划用尽时的整体完成检查——**执行者常常忘记说 done**（2026-09-28 真机实测：
        任务其实做完了，却因为执行者不吭声而整单判失败）。这里让评估者依据全部证据再判一次。"""
        lines = []
        for s in trace[-8:]:
            ev = s.evidence or {}
            lines.append(f"#{s.index} 目标「{s.goal}」动作 {s.cap}:{s.action}"
                         f" → 证据 ok={ev.get('ok')} output={str(ev.get('output'))[:200]!r}"
                         f" error={str(ev.get('error'))[:120]!r}")
        user = (f"任务：{task}\n\n执行轨迹与原始证据：\n" + "\n".join(lines) +
                "\n\n只依据上面这些原始证据判断：**整个任务**是否已经完成？"
                "证据不足以证明完成就判 ok=false。")
        return await self._call("evaluator", Verdict, SYS_EVALUATOR, user)

    async def supervise(self, task, trace):
        lines = [f"#{s.index} 目标「{s.goal}」动作 {s.cap}:{s.action} → {'成功' if s.ok else '失败'}（{s.reason[:80]}）"
                 for s in trace[-6:]]
        user = (f"任务：{task}\n最近的执行轨迹：\n" + "\n".join(lines) +
                "\n\n请判断是重规划还是放弃。")
        return await self._call("supervisor", Replan, SYS_SUPERVISOR, user)

    # ---------- 辅助 ----------
    def _device_brief(self, device):
        if not device:
            return "（未指定设备）"
        root = (getattr(device, "tags", None) or {}).get("root") or ""
        jail = f"，可写目录 {root}（文件路径必须在此目录内，相对路径按它解析）" if root else ""
        return (f"{device.device_id}（OS {device.os or '?'}，能力 {', '.join(device.capabilities) or '无'}，"
                f"阶段 {device.stage or '?'}{jail}）")

    def _evidence_text(self, evidence):
        if not evidence:
            return "（无证据：这一步还没执行）"
        clip = int(self.conf.get("evidence_clip") or 2000)
        out = evidence.get("output") or ""
        return json.dumps({
            "ok": evidence.get("ok"),
            "output": (out[:clip] + f"…(+{len(out) - clip}字)" if len(out) > clip else out),
            "error": evidence.get("error", ""),
            "extra": {k: v for k, v in (evidence.get("extra") or {}).items() if k != "screenshot"},
            "screenshot": ("有截图" if (evidence.get("extra") or {}).get("screenshot") else "无截图"),
        }, ensure_ascii=False)

    def _pick_device(self, device_id):
        want = device_id or self.conf.get("device_id") or ""
        if want:
            dev = next((d for d in self.hub.devices() if d["device_id"] == want), None)
            if dev is None:
                raise CuaError(f"执行器「{want}」不在线（已知：{', '.join(d['device_id'] for d in self.hub.devices()) or '无'}）"
                               "——目标 PC 上要先跑 executor_agent.py")
            return want
        online = self.hub.devices(only_online=True)
        if not online:
            raise CuaError("没有任何在线执行器——目标 PC 上先跑：python executor_agent.py --center <中心地址> --token <KEY>")
        return online[0]["device_id"]

    async def _dispatch(self, device_id, action, timeout=None):
        """派动作给执行器（hub 是同步阻塞的，放线程里跑，别卡住事件循环）。"""
        return await asyncio.to_thread(self.hub.dispatch, device_id, action.cap, action.action,
                                       action.args, timeout, False)

    def _evidence_from(self, result):
        return {"ok": result.ok, "output": result.output, "error": result.error,
                "extra": dict(result.extra or {}), "status": result.status}

    # ---------- 主流程 ----------
    async def run(self, task, device_id=None, max_steps=None):
        task = str(task or "").strip()
        if not task:
            raise CuaError("任务不能为空")
        if not self.conf.get("enabled", True):
            raise CuaError("Computer Use 循环被配置禁用（cua.enabled: false）")
        device_id = self._pick_device(device_id)
        dev = next((d for d in self.hub.devices() if d["device_id"] == device_id), None)
        max_steps = int(max_steps or self.conf.get("max_steps") or 12)
        max_failures = int(self.conf.get("max_failures") or 2)
        max_replans = int(self.conf.get("max_replans") or 0)

        t0 = time.time()
        from .executor_hub import ExecutorInfo

        dev_obj = ExecutorInfo(device_id, host=(dev or {}).get("host", ""), os_name=(dev or {}).get("os", ""),
                               capabilities=(dev or {}).get("capabilities"), stage=(dev or {}).get("stage"),
                               tags=(dev or {}).get("tags"))     # tags 里有可写目录 jail，必须带给模型
        plan = await self.make_plan(task, dev_obj)
        steps = list(plan.steps)[:max_steps]
        trace, replans, failures = [], 0, 0
        evidence = {"ok": None, "output": "", "error": "", "extra": {}}
        idx = 0

        while idx < max_steps:
            if not steps:
                # 计划用尽：先让评估者按「全部证据」判一次整体完成（执行者可能忘了说 done），
                # 确认没完成才交给监督者重规划（A25：监督者就是干这个的）
                if trace:
                    verdict = await self.judge_completion(task, trace)
                    if verdict.ok:
                        return self._finish(task, device_id, True,
                                            f"计划用尽，评估者依据全部证据确认任务已完成：{verdict.reason}",
                                            trace, replans, t0)
                if replans >= max_replans:
                    return self._finish(task, device_id, False,
                                        f"计划已用尽且重规划已用尽（{replans}/{max_replans}）",
                                        trace, replans, t0)
                try:
                    replan = await self.supervise(task, trace)
                except CuaError as e:
                    return self._finish(task, device_id, False, f"监督者不可用：{e}", trace, replans, t0)
                replans += 1
                failures = 0
                if replan.give_up:
                    return self._finish(task, device_id, False, f"监督者判定不可完成：{replan.reason}",
                                        trace, replans, t0)
                steps = list(replan.steps)[:max(0, max_steps - idx)]
                if not steps:
                    return self._finish(task, device_id, False, "监督者没有给出新步骤", trace, replans, t0)
                continue
            step = steps.pop(0)
            idx += 1
            action = await self.choose_action(task, step, evidence, dev_obj)
            if action.cap == "done":
                verdict = await self.judge(task, step.goal, action, evidence)
                trace.append(CuaStep(idx, step.goal, "done", "", {}, verdict.ok, verdict.reason, evidence))
                if verdict.ok:
                    return self._finish(task, device_id, True, f"执行者判定完成，评估者依据证据确认：{verdict.reason}",
                                        trace, replans, t0)
                failures += 1
            else:
                if self.conf.get("screenshot_each_step", True) and "screenshot" in (dev or {}).get("capabilities", []):
                    try:
                        await self._dispatch(device_id, Action(cap="screenshot", action="", args={}))
                    except Exception:
                        pass                                   # 截图拿不到不算失败，只是证据少一条
                try:
                    result = await self._dispatch(device_id, action)
                    evidence = self._evidence_from(result)
                except ExecutorError as e:
                    evidence = {"ok": False, "output": "", "error": f"{e.code}: {e.message}", "extra": {}}
                verdict = await self.judge(task, step.goal, action, evidence)
                trace.append(CuaStep(idx, step.goal, action.cap, action.action, action.args,
                                     verdict.ok, verdict.reason, evidence))
                if verdict.ok:
                    failures = 0
                    continue
                failures += 1

            if failures >= max_failures:
                if replans >= max_replans:
                    return self._finish(task, device_id, False,
                                        f"连续 {failures} 次未达成且重规划已用尽（{replans}/{max_replans}）",
                                        trace, replans, t0)
                try:
                    replan = await self.supervise(task, trace)
                except CuaError as e:
                    return self._finish(task, device_id, False, f"监督者不可用：{e}", trace, replans, t0)
                replans += 1
                failures = 0
                if replan.give_up:
                    return self._finish(task, device_id, False, f"监督者判定不可完成：{replan.reason}",
                                        trace, replans, t0)
                steps = list(replan.steps)[:max(0, max_steps - idx)]
        reason = "步数用尽仍未完成" if idx >= max_steps else "没有更多步骤"
        last = trace[-1].reason if trace else ""
        return self._finish(task, device_id, False, f"{reason}（最后一步评估：{last}）", trace, replans, t0)

    def _finish(self, task, device_id, ok, reason, trace, replans, t0):
        res = CuaResult(task, device_id, ok=ok, reason=reason, steps=trace, replans=replans,
                        ms=(time.time() - t0) * 1000, roles=dict(self.roles))
        try:
            from .logging_setup import log_event

            log_event("INFO" if ok else "WARNING", "cua_result", task=task[:200], device=device_id,
                      ok=ok, steps=len(trace), replans=replans, reason=reason[:200],
                      ms=round(res.ms, 1))
        except Exception:
            pass
        return res

    def run_sync(self, task, device_id=None, max_steps=None):
        """同步入口（CLI / 工具用）。"""
        return asyncio.run(self.run(task, device_id=device_id, max_steps=max_steps))


_LOCK = threading.Lock()
_CUA = None


def get_cua(cfg=None, hub=None, refresh=False) -> ComputerUseLoop:
    global _CUA
    with _LOCK:
        if _CUA is None or refresh:
            _CUA = ComputerUseLoop(cfg, hub=hub)
        return _CUA


def reset_cua():
    global _CUA
    with _LOCK:
        _CUA = None
