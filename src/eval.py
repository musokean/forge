"""评估模块（#13）：黄金集回归 + LLM-as-judge，防止 forge 改 prompt/配置后「变笨」。

对应 A10「防变笨」：每次对 forge 做结构性改动（prompt / 角色 / 模型 / 路由 / 技能）后，
跑一遍黄金集——一组标准问答对，验证核心能力没有退化。

判定双通道（互补）：
  ① 关键词命中（程序化硬指标）：每个用例声明期望答案必须包含的关键词，不依赖 LLM，断网可跑；
  ② LLM-as-judge（软指标）：复用 reflect.evaluate_answer 的评审角色给答案打分（0-10），
     低于用例 min_score 判失败；评审不可用时（返回 None）不扣分，只认关键词通道。

黄金集存 config/golden.yaml（配置驱动，/eval add 可扩展），报告支持 Markdown 导出。
"""

from __future__ import annotations
import asyncio
import datetime
import os
import time
from dataclasses import dataclass, field

import yaml

from .config import _BASE_DIR, load_config
from .console import C, paint
from .reflect import evaluate_answer

# 黄金集文件路径（相对项目根）
GOLDEN_PATH = os.path.join(_BASE_DIR, "config", "golden.yaml")
# 报告导出目录
EVAL_EXPORT_DIR = os.path.join(_BASE_DIR, "exports", "eval")

# 内置默认黄金集（首次运行无 golden.yaml 时自动写入；/eval add 追加）
DEFAULT_GOLDEN = [
    {"task": "计算 17 × 23 等于多少？", "keywords": ["391"], "min_score": 6,
     "note": "基础算术（calculator 工具）"},
    {"task": "1 公里等于多少米？", "keywords": ["1000"], "min_score": 6,
     "note": "常识换算，验证知识不退化"},
    {"task": "列出当前工作目录下的文件和子目录。", "keywords": ["main.py", "src"], "min_score": 6,
     "note": "文件系统工具（list_files）"},
    {"task": "用一句话介绍 forge 是什么。", "keywords": ["AI", "助手", "模型"], "min_score": 6,
     "note": "自我认知（system prompt 里的定位不跑偏）"},
    {"task": "在知识库里搜索「自动化」相关内容。", "keywords": ["自动任务", "调度"], "min_score": 5,
     "note": "知识库检索（kb_search，需知识库有相关内容，失败不判错）"},
    {"task": "今天的日期是什么？", "keywords": [str(datetime.date.today().year)], "min_score": 5,
     "note": "时间工具（get_time），动态断言当前年份"},
]


def _keyword_hit(answer: str, keyword: str) -> bool:
    """关键词命中判断：大小写不敏感子串匹配。"""
    return keyword.lower() in (answer or "").lower()


@dataclass
class EvalResult:
    """单个黄金用例的评估结果。"""
    task: str
    answer: str = ""
    keywords: list = field(default_factory=list)
    hits: list = field(default_factory=list)      # 每个关键词是否命中
    min_score: int = 6                             # judge 最低分
    judge_score: int | None = None                 # LLM-as-judge 打分（评审失败为 None）
    judge_issues: list = field(default_factory=list)
    passed: bool = False
    ms: float = 0.0                                # 单例耗时（毫秒）
    tokens: dict = field(default_factory=dict)
    note: str = ""

    @property
    def keyword_pass(self):
        """关键词通道：全部命中才算过（程序化硬指标）。"""
        return bool(self.keywords) and all(self.hits)

    @property
    def judge_pass(self):
        """judge 通道：评审不可用（None）视为通过（断网/评审失败不误杀），有分则须 ≥ min_score。"""
        return self.judge_score is None or self.judge_score >= self.min_score

    def verdict(self):
        return "✅" if self.passed else "❌"


class Evaluator:
    """黄金集加载 / 执行 / 报告。"""

    def __init__(self, golden_path: str | None = None):
        self.golden_path = golden_path or GOLDEN_PATH

    # ---------- 黄金集管理 ----------

    def load_golden(self) -> list[dict]:
        """加载黄金集；文件不存在则写入内置默认集再返回。"""
        if not os.path.exists(self.golden_path):
            self.save_golden(DEFAULT_GOLDEN)
        with open(self.golden_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cases = data.get("golden") or []
        # 规范化：补齐缺失字段
        for c in cases:
            c.setdefault("keywords", [])
            c.setdefault("min_score", 6)
            c.setdefault("note", "")
        return cases

    def save_golden(self, cases: list[dict]):
        """把黄金集写回 yaml（保注释风格：顶部说明 + golden 列表）。"""
        os.makedirs(os.path.dirname(self.golden_path), exist_ok=True)
        body = (
            "# ==============================================================================\n"
            "# forge · 黄金集（#13 评估 / A10 防变笨）\n"
            "# ------------------------------------------------------------------------------\n"
            "# 每次对 forge 做结构性改动（prompt / 角色 / 模型 / 路由 / 技能）后，跑 /eval 回归：\n"
            "#   · task      ：标准问答任务（forge 将真实执行，含工具调用）\n"
            "#   · keywords  ：期望答案必须包含的关键词（程序化硬指标，全命中才算过）\n"
            "#   · min_score ：LLM-as-judge 评审最低分 0-10（评审不可用时只看关键词）\n"
            "#   · note      ：备注（这例验证什么能力）\n"
            "# 添加用例：/eval add 任务|关键词1,关键词2|最低分，或直接编辑本文件。\n"
            "# ==============================================================================\n"
        )
        lines = [body, "golden:\n"]
        for c in cases:
            note = c.get("note", "")
            if note:
                lines.append(f"  # {note}\n")
            lines.append(f"  - task: {c['task']!r}\n")
            kw = ", ".join(repr(k) for k in c.get("keywords", []))
            lines.append(f"    keywords: [{kw}]\n")
            lines.append(f"    min_score: {c.get('min_score', 6)}\n")
            lines.append("\n")
        with open(self.golden_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)

    def add_case(self, task: str, keywords: list[str], min_score: int = 6, note: str = "") -> tuple[bool, str]:
        """新增黄金用例（写回 yaml）。返回 (ok, msg)。"""
        if not task.strip():
            return False, "任务不能为空"
        cases = self.load_golden()
        cases.append({"task": task.strip(), "keywords": keywords, "min_score": min_score, "note": note})
        self.save_golden(cases)
        return True, f"已添加黄金用例 #{len(cases)}：{task.strip()[:40]}"

    # ---------- 执行 ----------

    def _new_agent(self):
        """构造跑批专用 Agent（非流式、写操作放行、无 spinner）。子类/测试可覆盖。"""
        from .agent import Agent
        from .approval import Approver
        return Agent(stream=False, approver=Approver(auto_approve=True), show_spinner=False)

    async def run_case(self, case: dict, agent=None) -> EvalResult:
        """跑单个黄金用例：Agent 回答 → 关键词命中 → LLM-as-judge 打分。

        agent 不传则新建（非流式、写操作放行、无 spinner——回归跑批不需要交互审批）。
        """
        if agent is None:
            agent = self._new_agent()
        t0 = time.monotonic()
        answer = await agent.run(case["task"])
        ms = (time.monotonic() - t0) * 1000

        keywords = case.get("keywords") or []
        hits = [_keyword_hit(answer, k) for k in keywords]
        min_score = int(case.get("min_score", 6))

        # LLM-as-judge：评审角色打分；失败返回 None（不扣分，交给关键词通道判）
        judge_score, issues = None, []
        try:
            cfg = load_config()
            verdict = await evaluate_answer(cfg, case["task"], answer,
                                            judge_role=(cfg.get("reflect") or {}).get("judge_role", "fallback"))
            if verdict is not None:
                judge_score = verdict["score"]
                issues = verdict.get("issues") or []
        except Exception:
            judge_score = None  # 评审链路任何失败都不阻塞回归

        kw_pass = bool(keywords) and all(hits)
        judge_pass = judge_score is None or judge_score >= min_score
        return EvalResult(
            task=case["task"],
            answer=answer,
            keywords=keywords,
            hits=hits,
            min_score=min_score,
            judge_score=judge_score,
            judge_issues=issues,
            passed=kw_pass and judge_pass,
            ms=ms,
            tokens=dict(agent.total_tokens),
            note=case.get("note", ""),
        )

    async def run_all(self, cases: list[dict] | None = None, concurrency: int = 3,
                      agent_factory=None) -> list[EvalResult]:
        """并发跑全量黄金集（信号量限流，防撞上游限流）。

        agent_factory: 可选，无参可调用返回 Agent（测试注入 mock）；默认 None 用 _new_agent。
        """
        cases = cases or self.load_golden()
        sem = asyncio.Semaphore(concurrency)

        async def one(c):
            async with sem:
                return await self.run_case(c, agent=agent_factory() if agent_factory else None)

        return list(await asyncio.gather(*[one(c) for c in cases]))

    # ---------- 报告 ----------

    def report(self, results: list[EvalResult], title: str = "") -> str:
        """人类可读的回归报告（控制台输出用）。"""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_ms = sum(r.ms for r in results) / max(1, total)
        lines = [
            paint("■ 黄金集回归（#13 评估）", C.LIGHT_BLUE + C.BOLD)
            + paint(f"  {passed}/{total} 通过 · 平均 {avg_ms:.0f}ms/例", C.DIM),
            "",
        ]
        for i, r in enumerate(results, 1):
            kw = f"关键词 {sum(r.hits)}/{len(r.keywords)}" if r.keywords else "无关键词"
            judge = f"judge {r.judge_score}/10" if r.judge_score is not None else "judge —"
            verdict = paint(r.verdict(), C.SKY if r.passed else C.RED)
            lines.append(f"  {verdict} [{i}] {r.task[:36]}")
            lines.append(paint(f"       {kw} · {judge} · {r.ms:.0f}ms", C.DIM))
            if r.judge_issues and not r.passed:
                lines.append(paint(f"       ⚠ {r.judge_issues[0][:60]}", C.SKY))
        if title:
            lines.insert(0, paint(title, C.DIM))
        return "\n".join(lines)

    def export_markdown(self, results: list[EvalResult], name: str = "") -> str:
        """把回归报告导出为 Markdown（exports/eval/），返回文件路径。"""
        os.makedirs(EVAL_EXPORT_DIR, exist_ok=True)
        fname = name or f"eval-{datetime.datetime.now():%Y%m%d-%H%M%S}.md"
        path = os.path.join(EVAL_EXPORT_DIR, fname)
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        lines = [
            "# forge 黄金集回归报告（#13 评估）",
            "",
            f"- 时间：{datetime.datetime.now():%Y-%m-%d %H:%M}",
            f"- 结果：**{passed}/{total} 通过**",
            f"- 平均耗时：{sum(r.ms for r in results) / max(1, total):.0f} ms/例",
            "",
            "| # | 任务 | 关键词命中 | Judge 分 | 耗时 | 结论 |",
            "|---|------|-----------|---------|------|------|",
        ]
        for i, r in enumerate(results, 1):
            kw = f"{sum(r.hits)}/{len(r.keywords)}" if r.keywords else "—"
            judge = str(r.judge_score) if r.judge_score is not None else "—"
            verdict = "✅ 通过" if r.passed else "❌ 未过"
            lines.append(f"| {i} | {r.task[:40]} | {kw} | {judge} | {r.ms:.0f}ms | {verdict} |")
        lines += ["", "## 未通过详情", ""]
        for i, r in enumerate(results, 1):
            if r.passed:
                continue
            lines += [
                f"### [{i}] {r.task}",
                "",
                f"- 关键词命中：{sum(r.hits)}/{len(r.keywords)}（{'、'.join(r.keywords) if r.keywords else '—'}）",
                f"- Judge 评审：{r.judge_score}/10" if r.judge_score is not None else "- Judge 评审：不可用",
                f"- 耗时：{r.ms:.0f}ms",
                "",
                "**forge 的回答：**",
                "",
                "```",
                r.answer[:500],
                "```",
                "",
            ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path
