"""反思自纠错（Reflection / Self-Refine）：答案生成后由评审角色审视，低分则带意见重答。

对应 A07 可靠性组合拳「重试 + 降级 + 熔断 + 自纠错」的最后一环：
前三个都是「调用链」层面的保护，自纠错是「答案质量」层面的保护——
生成 → 评审（便宜模型打分）→ 低于阈值带具体意见重答一轮 → 仍低分则接受现状（不无限烧钱）。

配置（models.yaml reflect 段）：
  enabled: true        # 开关
  min_score: 6         # 0-10，低于此触发修正
  max_rounds: 1        # 最多修正轮数（每轮 = 评审 1 次 + 重答 1 次）
  judge_role: fallback # 评审角色（默认用降级便宜模型）
"""
import json
import re

from .llm import chat


async def evaluate_answer(cfg, task: str, answer: str, judge_role: str = "fallback"):
    """评审答案质量，返回 {"score": 0-10, "issues": [...], "suggestion": "..."}；任何失败返回 None（不阻塞）。"""
    msgs = [
        {"role": "system", "content": (
            "你是严格的答案评审。逐项检查：是否直接回答了问题、事实是否准确、"
            "是否遗漏关键信息、是否含废话或错误。只输出一个 JSON 对象："
            '{"score": 0到10的整数, "issues": ["问题1", "问题2"], "suggestion": "一句可执行的修正建议"}。'
            "不要输出任何其他内容。"
        )},
        {"role": "user", "content": f"问题：{task}\n\n答案：{answer[:4000]}"},
    ]
    try:
        resp = await chat(cfg, msgs, role=judge_role)
        content = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return None  # 模型没按约定输出 JSON：不评分（避免乱输出反而满分）
        d = json.loads(m.group(0))
        score = int(d.get("score", -1))
        if not (0 <= score <= 10):
            return None
        return {
            "score": score,
            "issues": d.get("issues") or [],
            "suggestion": d.get("suggestion") or "",
        }
    except Exception:
        return None  # 评审失败静默跳过（不阻塞主流程）


async def refine_answer(cfg, task: str, answer: str, feedback: str, role: str = "default"):
    """带评审意见重答一次（非流式取完整答案）；失败返回原答案。"""
    msgs = [
        {"role": "system", "content": (
            "你是 forge（匠），一个多模型驱动的 AI 助手。请根据评审意见修正你上一版回答："
            "保留正确的部分，补齐遗漏，纠正错误。直接输出修正后的完整答案，不要解释修改过程。"
        )},
        {"role": "user", "content": f"原问题：{task}\n\n你的上一版答案：{answer[:4000]}\n\n评审意见：{feedback}\n\n请输出修正后的完整答案。"},
    ]
    try:
        resp = await chat(cfg, msgs, role=role)
        new = (resp.choices[0].message.content or "").strip()
        return new or answer
    except Exception:
        return answer  # 修正失败保留原答案（自纠错绝不比不纠更差）
