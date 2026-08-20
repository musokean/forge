"""配置写回：在 CLI 里安全修改 config/models.yaml。

关键约束：**逐行文本替换**，只改用户选中的那一个值，其余整份内容（包括
那堆中文注释、切回 DeepSeek 的模板、对齐空格）原样保留——绝不用 yaml.dump
重写整个文件（那会把注释全丢光）。零外部依赖，只用标准库 re/os。

对外提供 4 个 setter（都返回 (ok: bool, msg: str)，带存在性校验）：
  · set_role_model(role, alias)     改「角色 → 模型别名」
  · set_debate_model(name, role)    改「辩手 → 角色」
  · set_debate_rounds(n)            改辩论轮数
  · set_router_role(role)           改自动路由判断角色
"""
import os
import re

from .config import _BASE_DIR, load_config, _models_list


def config_path():
    return os.path.join(_BASE_DIR, "config", "models.yaml")


def _read():
    with open(config_path(), "r", encoding="utf-8") as f:
        return f.read().splitlines(keepends=True)


def _write(lines):
    # newline="" 保留各行原有换行符，避免整份文件被改成别的换行风格
    with open(config_path(), "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)


def _is_top_key(line):
    """顶格（无缩进）的 YAML key，如 `roles:`；注释行 `# ...` 不算。"""
    return bool(re.match(r"^[A-Za-z_]\w*\s*:", line))


def _section_range(lines, top_key):
    """返回顶格段 top_key 的内容行区间 [start, end)（不含 `top_key:` 那一行本身）。

    段结束于下一个顶格 key；中间的空行/注释都算段内。
    """
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^{re.escape(top_key)}\s*:", ln):
            start = i + 1
            break
    if start is None:
        return None, None
    end = len(lines)
    for j in range(start, len(lines)):
        if _is_top_key(lines[j]):
            end = j
            break
    return start, end


def _split_nl(line):
    """拆出「正文」与「行尾换行符」，替换正文时保留原换行。"""
    body = line.rstrip("\r\n")
    return body, line[len(body):]


# ---------- 只读辅助：给 UI 列可选项 ----------

def available_model_aliases(cfg=None):
    cfg = cfg or load_config()
    return [name for name, _ in _models_list(cfg)]


def available_roles(cfg=None):
    cfg = cfg or load_config()
    return list((cfg.get("roles") or {}).keys())


# ---------- 4 个 setter ----------

def set_role_model(role, alias):
    """改 roles.<role> 指向的模型别名。兼容两种写法：
       default: { model: X, label: .., purpose: .. }   # 行内 dict
       default: X                                        # 简写
    """
    cfg = load_config()
    aliases = available_model_aliases(cfg)
    if alias not in aliases:
        return False, f"模型别名「{alias}」不存在，可选：{', '.join(aliases)}"
    if role not in available_roles(cfg):
        return False, f"角色「{role}」不存在，可选：{', '.join(available_roles(cfg))}"

    lines = _read()
    s, e = _section_range(lines, "roles")
    if s is None:
        return False, "配置里找不到 roles 段"

    pat = re.compile(rf"^(\s+){re.escape(role)}(\s*:\s*)(.*)$")
    for i in range(s, e):
        body, nl = _split_nl(lines[i])
        mm = pat.match(body)
        if not mm:
            continue
        indent, colon, rest = mm.group(1), mm.group(2), mm.group(3)
        if rest.lstrip().startswith("{"):
            # 行内 dict：只替换第一个 model: xxx，label/purpose 原样保留
            new_rest = re.sub(r"(model\s*:\s*)([^,}\s]+)", rf"\g<1>{alias}", rest, count=1)
        else:
            # 简写：把别名换掉，保留可能存在的行尾注释
            m2 = re.match(r"^(\s*)([^\s#]+)(.*)$", rest)
            new_rest = f"{m2.group(1)}{alias}{m2.group(3)}" if m2 else alias
        lines[i] = f"{indent}{role}{colon}{new_rest}{nl}"
        _write(lines)
        return True, f"角色「{role}」→ 模型「{alias}」"
    return False, f"roles 段里没找到角色「{role}」的定义行"


def set_debate_model(debater_name, role):
    """改 debate.roles 里某个辩手（按 name 定位）绑定的角色。"""
    cfg = load_config()
    roles = available_roles(cfg)
    if role not in roles:
        return False, f"角色「{role}」不存在，可选：{', '.join(roles)}"

    lines = _read()
    s, e = _section_range(lines, "debate")
    if s is None:
        return False, "配置里找不到 debate 段"

    name_pat = re.compile(r"^\s*-\s*name\s*:\s*([^\s#]+)")
    model_pat = re.compile(r"^(\s*model\s*:\s*)([^\s#]+)(.*)$")
    current = None
    for i in range(s, e):
        body, nl = _split_nl(lines[i])
        nm = name_pat.match(body)
        if nm:
            current = nm.group(1)
            continue
        if current == debater_name:
            mm = model_pat.match(body)
            if mm:
                lines[i] = f"{mm.group(1)}{role}{mm.group(3)}{nl}"
                _write(lines)
                return True, f"辩手「{debater_name}」→ 角色「{role}」"
    return False, f"debate 段里没找到辩手「{debater_name}」的 model 行"


def set_debate_rounds(n):
    """改辩论轮数（1~10）。"""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return False, "轮数得是整数"
    if not (1 <= n <= 10):
        return False, "轮数建议 1~10"

    lines = _read()
    s, e = _section_range(lines, "debate")
    if s is None:
        return False, "配置里找不到 debate 段"

    pat = re.compile(r"^(\s*rounds\s*:\s*)(\d+)(.*)$")
    for i in range(s, e):
        body, nl = _split_nl(lines[i])
        mm = pat.match(body)
        if mm:
            lines[i] = f"{mm.group(1)}{n}{mm.group(3)}{nl}"
            _write(lines)
            return True, f"辩论轮数 → {n}"
    return False, "debate 段里没有 rounds 行"


def set_router_role(role):
    """改自动路由用哪个角色做任务类型判断。"""
    cfg = load_config()
    roles = available_roles(cfg)
    if role not in roles:
        return False, f"角色「{role}」不存在，可选：{', '.join(roles)}"

    lines = _read()
    s, e = _section_range(lines, "router")
    if s is None:
        return False, "配置里找不到 router 段"

    pat = re.compile(r"^(\s*role\s*:\s*)([^\s#]+)(.*)$")
    for i in range(s, e):
        body, nl = _split_nl(lines[i])
        mm = pat.match(body)
        if mm:
            lines[i] = f"{mm.group(1)}{role}{mm.group(3)}{nl}"
            _write(lines)
            return True, f"自动路由判断角色 →「{role}」"
    return False, "router 段里没有 role 行"


DEFAULT_DEBATE = """\
debate:
  rounds: 2
  roles:
    - name: 正方
      model: default
      persona: 你代表正方立场，坚定论证这个观点/方案的合理性，找出并强调它的优点与价值。
    - name: 反方
      model: default
      persona: 你代表反方立场，质疑并找出这个观点/方案的漏洞、风险与代价，客观反驳。
    - name: 裁判
      model: default
      persona: 你是中立的裁判，不偏袒任何一方，综合双方观点，给出客观、平衡、可执行的最终结论。
"""


def setup_debate_defaults(roles_to_use=("正方", "反方", "裁判")):
    """一键配置默认辩论阵容（老大 2026-08-20：辩论角色默认不配，有需要引导配置）。

    把 debate 段的 `roles: []`（或空列表）替换成默认三辩手；默认都绑 default 角色
    （后续可在 /config 里分别换角色）。返回 (ok, msg)。
    """
    cfg = load_config()
    roles = available_roles(cfg)
    if "default" not in roles:
        return False, "需要至少一个「default」角色才能配置辩论"
    lines = _read()
    s, e = _section_range(lines, "debate")
    if s is None:
        return False, "配置里找不到 debate 段"

    # 找 `roles: []` 行，替换为默认三辩手（缩进 2 空格）
    for i in range(s, e):
        body, nl = _split_nl(lines[i])
        if re.match(r"^\s*roles\s*:\s*\[\s*\]\s*(#.*)?$", body):
            default_block = "debate:\n  rounds: 2\n  roles:\n" + "".join(
                f"    - name: {name}\n      model: default\n      persona: {persona}\n"
                for name, persona in zip(
                    ("正方", "反方", "裁判"),
                    (
                        "你代表正方立场，坚定论证这个观点/方案的合理性，找出并强调它的优点与价值。",
                        "你代表反方立场，质疑并找出这个观点/方案的漏洞、风险与代价，客观反驳。",
                        "你是中立的裁判，不偏袒任何一方，综合双方观点，给出客观、平衡、可执行的最终结论。",
                    ),
                )
            )
            # 段起点：`debate:` 行在 s-1（_section_range 不含它本身）
            seg_start = s - 1
            if seg_start < 0 or not re.match(r"^debate\s*:", _split_nl(lines[seg_start])[0]):
                return False, "debate 段格式异常"
            lines[seg_start : i + 1] = [default_block]
            _write(lines)
            return True, "已配置默认辩论阵容：正方 / 反方 / 裁判（都绑 default，可在 /config 换角色）"
    return False, "debate 段里没找到空的 roles: [] 行（可手动把 roles 清空后重试）"


def set_kb_path(path):
    """改知识库索引库路径（knowledge.db_path，可相对项目根或绝对路径）。"""
    path = path.strip()
    if not path:
        return False, "路径不能为空"
    if not path.endswith(".db"):
        return False, "建议以 .db 结尾（SQLite 库文件）"

    lines = _read()
    s, e = _section_range(lines, "knowledge")
    if s is None:
        return False, "配置里找不到 knowledge 段"

    pat = re.compile(r"^(\s*db_path\s*:\s*)([^\s#]+)(.*)$")
    for i in range(s, e):
        body, nl = _split_nl(lines[i])
        mm = pat.match(body)
        if mm:
            lines[i] = f"{mm.group(1)}{path}{mm.group(3)}{nl}"
            _write(lines)
            return True, f"知识库路径 →「{path}」（重启或 /kb 操作后生效）"
    return False, "knowledge 段里没有 db_path 行"


def set_model_key(alias, value, use_env=False):
    """给某个模型配置 key：明文写 api_key，或 env: 前缀写 api_key_env。

    预置主流模型默认只有 api_key_env 模板行；选中后输入明文 key 时，
    会把原 api_key_env 行**原位替换**成 api_key: <key>，其余字段与注释不动。
    """
    cfg = load_config()
    aliases = available_model_aliases(cfg)
    if alias not in aliases:
        return False, f"模型别名「{alias}」不存在，可选：{', '.join(aliases)}"
    value = value.strip()
    if not value:
        return False, "key 不能为空"

    lines = _read()
    s, e = _section_range(lines, "models")
    if s is None:
        return False, "配置里找不到 models 段"

    alias_pat = re.compile(r"^  " + re.escape(alias) + r"\s*:$")
    key_pat = re.compile(r"^(\s{4})(api_key_env|api_key)(\s*:\s*)([^\s#]*)(.*)$")
    current = None
    for i in range(s, e):
        body, nl = _split_nl(lines[i])
        if alias_pat.match(body):
            current = alias
            continue
        if current != alias:
            continue
        # 本块结束：又遇到下一个 2 空格缩进的别名行
        if re.match(r"^  [A-Za-z_]\w*\s*:$", body):
            break
        km = key_pat.match(body)
        if km:
            field = "api_key_env" if use_env else "api_key"
            lines[i] = f"{km.group(1)}{field}{km.group(3)}{value}{km.group(5)}{nl}"
            _write(lines)
            return True, f"模型「{alias}」key 已配置（{'环境变量' if use_env else '明文'}）"
    return False, f"models 段里找不到「{alias}」的 api_key / api_key_env 行"
