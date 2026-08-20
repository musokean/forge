"""M3 #7 trace 基础：每角色/每步可观测（A09 可观测性补全）。

轻量 Tracer：内存事件流 + JSONL 导出。Agent.run() 全流程打点，
多智能体各自独立，可汇总打印——为 M5 完整日志打底。

事件类型：
  run_start  一次 run() 开始（任务文本）
  llm_step   一次模型调用完成（step 序号、token、耗时）
  tool       一次工具调用（名称、参数摘要、结果摘要、耗时）
  answer     返回最终答案（内容摘要）
  run_end    一次 run() 结束（总耗时、总 token）
"""
import json
import time


class Tracer:
    def __init__(self, agent_name="forge", role=None, model=None):
        self.agent = agent_name
        self.role = role
        self.model = model
        self.events = []  # [{t, type, ms, **fields}]

    # ---------- 记录 ----------

    def record(self, type_, **fields):
        """追加一条事件（自动带时间戳与耗时毫秒）。"""
        self.events.append({
            "t": time.strftime("%H:%M:%S"),
            "type": type_,
            "ms": fields.pop("ms", None),
            **fields,
        })

    # ---------- 汇总 / 输出 ----------

    def summary(self):
        """结构化摘要：步数、工具调用序列、token、耗时。"""
        steps = [e for e in self.events if e["type"] == "llm_step"]
        tools = [e for e in self.events if e["type"] == "tool"]
        run_end = next((e for e in reversed(self.events) if e["type"] == "run_end"), None)
        return {
            "agent": self.agent,
            "model": self.model,
            "steps": len(steps),
            "tools": [f"{e['name']}({e.get('arg_summary', '')[:30]})" for e in tools],
            "llm_ms": sum(e.get("ms") or 0 for e in steps),
            "tool_ms": sum(e.get("ms") or 0 for e in tools),
            "total_ms": (run_end or {}).get("ms"),
            "prompt_tokens": (run_end or {}).get("prompt_tokens"),
            "completion_tokens": (run_end or {}).get("completion_tokens"),
        }

    def one_line(self):
        """单行浓缩摘要（多智能体汇总打印用）。"""
        s = self.summary()
        tok = (s["prompt_tokens"] or 0) + (s["completion_tokens"] or 0)
        ms = s["total_ms"]
        return (f"{s['agent']}({s['model'] or '?'}): {s['steps']}步 "
                f"{len(s['tools'])}工具 {ms / 1000:.1f}s {tok}tok" if ms is not None
                else f"{s['agent']}({s['model'] or '?'}): {s['steps']}步 {len(s['tools'])}工具 {tok}tok")

    def render(self):
        """人类可读的流水文本（CLI /trace 用）。"""
        s = self.summary()
        lines = [
            f"  ── trace · {s['agent']} ({s['model'] or '?'}) ──",
            f"  模型步数 {s['steps']} ｜ 模型耗时 {s['llm_ms']:.0f}ms ｜ 工具 {len(s['tools'])} 次 {s['tool_ms']:.0f}ms"
            + (f" ｜ 总耗时 {s['total_ms']:.0f}ms" if s["total_ms"] is not None else ""),
        ]
        for i, e in enumerate(self.events):
            tag = e["type"]
            body = ""
            if tag == "run_start":
                body = (e.get("task") or "")[:50]
            elif tag == "llm_step":
                ms = e.get("ms")
                body = f"step{e.get('step')} · +{e.get('tokens') or 0} tok" + (f" · {ms:.0f}ms" if ms else "")
            elif tag == "tool":
                body = f"{e.get('name')}({(e.get('arg_summary') or '')[:40]}) → {(e.get('result_summary') or '')[:30]}"
            elif tag == "answer":
                body = (e.get("content") or "")[:50]
            lines.append(f"  [{e['t']}] {tag:9s} {body}")
        return "\n".join(lines)

    # ---------- 落盘 / 读取 ----------

    def to_jsonl(self, path):
        """导出为 JSONL（M5 完整日志的前身）。"""
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps({"agent": self.agent, **e}, ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(path):
        """从 JSONL 读取事件列表。"""
        events = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
