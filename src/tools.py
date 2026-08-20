"""工具系统：@tool 装饰器注册工具，生成 Function Calling 的 schema（对应 A02 工具调用、A06 只读分级）。"""
import ast
import datetime
import json
import math
import operator
import os
import re
import ssl
import subprocess
import urllib.parse
import urllib.request
from html import unescape
from typing import Callable

TOOLS = {}  # 工具注册表


def tool(name: str, description: str, parameters: dict, read_only: bool = True):
    """@tool 装饰器：注册一个可被 Agent 调用的工具。

    参数：
    - name: 工具名（模型看到的名字）
    - description: 工具用途（模型据此判断何时调用）
    - parameters: JSON Schema，描述入参
    - read_only: 是否只读（False 表示写操作/有副作用，对应 A06 只读分级）
    """
    def deco(fn: Callable):
        TOOLS[name] = {
            "fn": fn,
            "description": description,
            "parameters": parameters,
            "read_only": read_only,
        }
        return fn
    return deco


def is_write(name: str) -> bool:
    """判断工具是否写操作（有副作用，需审批/警惕）。"""
    spec = TOOLS.get(name)
    return spec is not None and not spec["read_only"]


def get_tools_schema():
    """生成 OpenAI 格式的 tools 参数（Function Calling schema）。"""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for name, spec in TOOLS.items()
    ]


def execute(name: str, arguments: dict):
    """执行工具，返回统一结构（对应 A07 错误处理）。"""
    spec = TOOLS.get(name)
    if not spec:
        return {"ok": False, "error": f"未知工具 {name}"}
    try:
        return {"ok": True, "data": spec["fn"](**arguments)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ============ 联网辅助（零依赖，标准库实现）============

def _http_get(url: str, timeout: int = 15) -> str:
    """urllib 抓取网页；沙箱 CA 缺失时降级为跳过证书校验。"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read().decode("utf-8", errors="ignore")


def _strip_html(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text or ""))


def _extract_text(html: str) -> str:
    """去掉 script/style 与标签，提取正文纯文本。"""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    text = _strip_html(html)
    return re.sub(r"\s+", " ", text).strip()


def _decode_bytes(b: bytes) -> str:
    """跨平台解码子进程输出（Windows 中文是 GBK，Linux 是 UTF-8）。"""
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


# ============ 安全表达式求值（calculator 用，禁 eval 防注入）============

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_SAFE_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow, "int": int, "float": float}
_SAFE_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_eval(expr: str):
    """只允许数字、四则运算、幂/取余、白名单函数与 pi/e 常量，拒绝任意代码。"""
    node = ast.parse(expr, mode="eval").body

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[n.func.id](*[ev(a) for a in n.args])
        if isinstance(n, ast.Name) and n.id in _SAFE_CONSTS:
            return _SAFE_CONSTS[n.id]
        raise ValueError(f"不支持的表达式")

    return ev(node)


# ============ 文件操作工具 ============

@tool(
    name="read_file",
    description="读取本地文本文件内容（超过 10 万字符自动截断），入参 path 为文件路径",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    read_only=True,
)
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read(100001)  # 多读 1 字符判断是否超长
    if len(content) > 100000:
        return content[:100000] + "\n\n[文件过大，已截断前 100000 字符]"
    return content


@tool(
    name="write_file",
    description="把 content 写入本地文件 path（写操作，有副作用）",
    parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    read_only=False,
)
def write_file(path: str, content: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}"


@tool(
    name="edit_file",
    description="局部修改文件：把文件里第一个 old_str 替换为 new_str（写操作）。入参 path 文件路径、old_str 原文、new_str 新文",
    parameters={"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}}, "required": ["path", "old_str", "new_str"]},
    read_only=False,
)
def edit_file(path: str, old_str: str, new_str: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old_str not in content:
        return "未找到要替换的内容，文件未修改"
    content = content.replace(old_str, new_str, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已修改 {path}"


@tool(
    name="list_files",
    description="列出目录下的文件和子目录。入参 path 为目录路径，recursive 为 True 时递归列出",
    parameters={"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]},
    read_only=True,
)
def list_files(path: str, recursive: bool = False) -> str:
    if not os.path.isdir(path):
        return f"路径不存在或不是目录：{path}"
    if recursive:
        entries = [os.path.join(root, f) for root, _, files in os.walk(path) for f in files]
    else:
        entries = [os.path.join(path, e) for e in os.listdir(path)]
    if not entries:
        return "（空目录）"
    return "\n".join(entries[:200])


@tool(
    name="search_file",
    description="在目录下搜索包含关键词的文本文件（类似 grep）。入参 path 为目录，pattern 为关键词",
    parameters={"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["path", "pattern"]},
    read_only=True,
)
def search_file(path: str, pattern: str) -> str:
    if not os.path.isdir(path):
        return f"路径不存在或不是目录：{path}"
    matches = []
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if pattern in line:
                            matches.append(f"{fp}:{i}: {line.strip()[:120]}")
                            if len(matches) >= 50:
                                return "\n".join(matches)
            except Exception:
                continue
    return "\n".join(matches) if matches else f"未找到包含「{pattern}」的内容"


# ============ 计算 / 命令 / 时间 / 联网工具 ============

@tool(
    name="calculator",
    description="安全计算数学表达式（支持 +-*/、括号、幂、取余、abs/round/min/max、pi/e）。入参 expression 为算式字符串",
    parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    read_only=True,
)
def calculator(expression: str) -> str:
    return str(_safe_eval(expression))


@tool(
    name="run_command",
    description="在本地执行一条 shell 命令并返回输出（有副作用，谨慎使用）。入参 command 为命令字符串",
    parameters={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    read_only=False,
)
def run_command(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True, timeout=30)
        out = _decode_bytes(r.stdout or b"") + _decode_bytes(r.stderr or b"")
        return out.strip() or "(命令无输出)"
    except subprocess.TimeoutExpired:
        return "命令超时（30 秒）"


@tool(
    name="get_time",
    description="获取当前日期和时间（含星期）。无入参",
    parameters={"type": "object", "properties": {}},
    read_only=True,
)
def get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S %A")


@tool(
    name="web_search",
    description="联网搜索，返回相关结果的标题、链接和摘要。入参 query 为搜索关键词",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    read_only=True,
)
def web_search(query: str) -> str:
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(query) + "&setlang=zh-hans&count=10"
    try:
        html = _http_get(url)
    except Exception as e:
        return f"搜索失败：{e}"
    items = []
    for block in re.findall(r'<li class="b_algo".*?</li>', html, re.S):
        a = re.search(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not a:
            continue
        link, title = a.group(1), _strip_html(a.group(2)).strip()
        p = re.search(r'<p[^>]*>(.*?)</p>', block, re.S)
        snippet = _strip_html(p.group(1)).strip() if p else ""
        if title and link.startswith("http"):
            items.append(f"- {title}\n  {link}" + (f"\n  {snippet}" if snippet else ""))
        if len(items) >= 5:
            break
    return "\n\n".join(items) if items else "未搜索到结果，或搜索引擎暂不可用。"


@tool(
    name="web_fetch",
    description="抓取指定 URL 的网页正文纯文本。入参 url 为完整网址（含 http/https）",
    parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    read_only=True,
)
def web_fetch(url: str) -> str:
    try:
        html = _http_get(url)
    except Exception as e:
        return f"抓取失败：{e}"
    text = _extract_text(html)
    return text[:3000] if text else "未提取到正文内容。"


# ============ 知识库工具（M3 #15 SQLite+FTS5） ============

_KB = None
_DEFAULT_KB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge.db")


def _kb_path():
    """知识库路径：优先 config/models.yaml 的 knowledge.db_path（配置驱动），缺省回退默认。"""
    try:
        from .config import load_config, _BASE_DIR  # 延迟导入避免模块级循环
        cfg = load_config()
        p = (cfg.get("knowledge") or {}).get("db_path")
        if p:
            return p if os.path.isabs(p) else os.path.join(_BASE_DIR, p)
    except Exception:
        pass
    return _DEFAULT_KB_PATH


def _get_kb():
    """知识库全局单例（懒加载，路径随配置）。"""
    global _KB
    if _KB is None:
        from .knowledge import KnowledgeBase  # 延迟导入避免模块级循环
        _KB = KnowledgeBase(_kb_path())
    return _KB


def reset_kb():
    """知识库路径配置变更后重置单例（下次 _get_kb 按新路径重建）。"""
    global _KB
    _KB = None


@tool(
    name="kb_search",
    description="在本地知识库中全文检索（已建索引的文档：md/txt/py/json/yaml 等文本）。入参 query 检索词（支持中文子串），limit 返回条数上限。查本地资料/历史文档时用，与 web_search 互补。",
    parameters={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
    read_only=True,
)
def kb_search(query: str, limit: int = 5) -> str:
    return json.dumps(_get_kb().query(query, limit), ensure_ascii=False)


@tool(
    name="kb_ingest",
    description="把本地文件或目录加入知识库索引（建索引后可被 kb_search 检索；写操作）。目录会递归收集文本文件，重复导入自动跳过未变化的文件。入参 path 为文件或目录路径",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    read_only=False,
)
def kb_ingest(path: str) -> str:
    added, skipped = _get_kb().ingest(path)
    return json.dumps({"ok": True, "added": added, "skipped": skipped, "stats": _get_kb().stats()}, ensure_ascii=False)


@tool(
    name="kb_add",
    description="把一条知识直接写入知识库（索引库即源文档，无需外部文件）：入参 title 标题、content 正文。用户要求「记到知识库 / 记住这个 / 沉淀要点」时用；标题重复会覆盖更新。写操作",
    parameters={"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]},
    read_only=False,
)
def kb_add(title: str, content: str) -> str:
    ok, msg = _get_kb().add(title, content)
    return json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False)
