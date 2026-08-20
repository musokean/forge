"""结构化输出（#6）：鲁棒 JSON 抽取 + Pydantic schema 校验 + 解析失败重试（对应 A08 / A27 结构化 handoff）。

核心能力：让模型稳定返回「符合 schema 的 JSON」而不是散文，解析失败就把错误喂回模型重试。
跨角色交接（并行/辩论）用 RoleBrief 这类 Pydantic 对象，而不是把整段对话史塞给下游。
"""
import json
import re

from pydantic import BaseModel, Field, ValidationError

from .llm import chat


class StructuredError(Exception):
    """结构化输出重试后仍失败。"""


def extract_json(text: str) -> dict:
    """从模型回复里鲁棒抽取 JSON 对象。

    处理：```json 围栏、前导/尾随散文、尾随逗号、单引号键。返回 dict。
    """
    if not text:
        raise ValueError("回复为空，无法抽取 JSON")
    s = text.strip()
    # 去 ```json ... ``` 围栏
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    # 截取第一个 { 到最后一个 }
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]
    # 去尾随逗号（如 {"a":1,}）
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # 退一步：单引号当双引号（模型偶尔用单引号）
        s2 = re.sub(r"(?<!\\)'", '"', s)
        return json.loads(s2)


def _schema_hint(schema: type[BaseModel]) -> str:
    """把 Pydantic schema 的字段说明渲染成引导文本。"""
    lines = []
    for name, field in schema.model_fields.items():
        req = "" if field.is_required() else "（可选）"
        desc = field.description or ""
        ann = getattr(field.annotation, "__name__", str(field.annotation))
        lines.append(f"- {name}{req}: {desc}（类型 {ann}）")
    return "\n".join(lines)


_SCHEMA_INSTR = (
    "你必须只输出一个 JSON 对象，不要任何解释、不要 Markdown 代码围栏。"
    "严格按照以下字段与类型输出：\n{fields}\n"
)


async def ask_structured(cfg, messages, schema: type[BaseModel], role="default", retries=2):
    """非流式调用模型，要求返回符合 schema 的 JSON；解析/校验失败把错误喂回重试。

    返回 (validated_obj, raw_resp)。
    不修改传入的 messages（指令与重试纠错都在内部副本上进行）。
    """
    fields = _schema_hint(schema)
    instr = _SCHEMA_INSTR.format(fields=fields)
    msgs = list(messages) + [{"role": "user", "content": instr}]
    last_err = None
    for _ in range(retries + 1):
        resp = await chat(cfg, msgs, role=role)
        content = resp.choices[0].message.content or ""
        try:
            data = extract_json(content)
            obj = schema.model_validate(data)
            return obj, resp
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            msgs = msgs + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    f"你的回复不是合法 JSON 或不符合要求的结构：{e}\n"
                    "请只输出符合上述字段要求的 JSON 对象，不要任何其他文字。"
                )},
            ]
    raise StructuredError(f"结构化输出重试 {retries} 次仍失败：{last_err}")


class RoleBrief(BaseModel):
    """跨角色交接的标准化摘要对象（A27 结构化 handoff）。

    并行/辩论里，每个角色把自己的产出压成这样一个对象交给下游，
    下游拿结构化字段，不用去解析上游的散文。
    """

    name: str = Field(description="角色名或任务名")
    stance: str = Field(default="", description="一句话立场或结论摘要")
    key_points: list[str] = Field(default_factory=list, description="关键要点列表")
    confidence: float = Field(default=0.5, description="置信度 0~1")
    source: str = Field(default="", description="信息来源或依据")
