"""Web 界面（#12，对应 A19）：零依赖 HTTP 聊天服务。

设计取舍：
  · 不引 FastAPI/Flask —— 保持项目「零重依赖」哲学，标准库 http.server 足够（单用户本地聊天）。
  · 非流式直答 —— POST /api/chat 一次性返回完整答案；前端等待动画兜底（体验等价 CLI 非流式）。
  · 写操作安全默认 —— Web 端无交互审批通道，Agent 用 auto_reject（写操作被拒并说明，
    引导回 CLI 执行）；只读工具（检索/计算/读文件/联网）完全可用。
  · 单会话 —— 一个全局 Agent 实例，/api/reset 重置上下文；多会话留给 M5 部署（#14）。

启动：forge --web [--port 8000]；或在 REPL 里 /web 拉起。
"""
import asyncio
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import Agent
from .approval import Approver
from .config import load_config, resolve_model
from .console import C, paint

PAGE_TITLE = "forge · 把想法锻造成现实"

# ---------- 内嵌聊天页面（单文件，零外部资源；浅蓝主题 #58a6ff） ----------
# 注意：页面里有大量 JS 花括号，绝不能 .format()——用 __TITLE__ 占位符 + replace

_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --blue: #58a6ff; --blue-deep: #2f81f7; --bg: #f0f4f9; --card: #ffffff;
    --text: #1f2328; --dim: #6e7681; --border: #d8dee6; --user-bg: #e8f1fe;
  }
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
         background: linear-gradient(160deg, #eef4fc 0%, var(--bg) 40%, #eaf0fa 100%);
         color: var(--text); height: 100vh; display: flex; flex-direction: column; }}
  header {{ background: rgba(255,255,255,.85); backdrop-filter: blur(6px);
           border-bottom: 1px solid var(--border); padding: 10px 20px;
           display: flex; align-items: center; gap: 12px; position: sticky; top: 0; z-index: 10; }}
  header .logo {{ color: var(--blue-deep); font-weight: 800; font-size: 20px; letter-spacing: 2px; }}
  header .slogan {{ color: var(--dim); font-size: 12px; }}
  header .status {{ margin-left: auto; color: var(--dim); font-size: 12px; font-variant-numeric: tabular-nums; }}
  #chat {{ flex: 1; overflow-y: auto; padding: 24px 20px; max-width: 880px; width: 100%; margin: 0 auto; scroll-behavior: smooth; }}
  .day {{ text-align: center; color: var(--dim); font-size: 11px; margin: 8px 0 18px; }}
  .msg {{ display: flex; margin-bottom: 18px; gap: 10px; }}
  .msg .avatar {{ width: 34px; height: 34px; border-radius: 50%; flex: none;
                 display: flex; align-items: center; justify-content: center; font-size: 17px;
                 background: linear-gradient(135deg, #cfe4ff, #a8cdfb); box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .msg.user {{ flex-direction: row-reverse; }}
  .msg.user .avatar {{ background: linear-gradient(135deg, #d9f0d0, #b8e2a9); }}
  .msg .body {{ max-width: 78%; display: flex; flex-direction: column; }}
  .msg .bubble {{ padding: 10px 14px; border-radius: 14px; line-height: 1.7; font-size: 14px;
                 white-space: pre-wrap; word-break: break-word; }}
  .msg.forge .bubble {{ background: var(--card); border: 1px solid var(--border);
                       border-top-left-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .msg.user .bubble {{ background: var(--user-bg); border: 1px solid #d2e5fd;
                      border-top-right-radius: 4px; }}
  .msg .meta {{ font-size: 11px; color: var(--dim); margin: 2px 4px; }}
  .msg.user .meta {{ text-align: right; }}
  .bubble code {{ background: #f3f5f8; border: 1px solid var(--border); border-radius: 4px;
                 padding: 1px 5px; font-family: Consolas, monospace; font-size: 12.5px; }}
  .bubble pre {{ background: #1f2428; color: #e6edf3; border-radius: 8px; padding: 10px 12px;
                overflow-x: auto; margin: 6px 0; }}
  .bubble pre code {{ background: transparent; border: none; color: inherit; padding: 0; }}
  .bubble strong {{ color: var(--blue-deep); }}
  .copy-btn {{ display: inline-block; margin: 4px 2px 0 0; padding: 2px 8px; font-size: 11px;
              color: var(--dim); background: transparent; border: 1px solid var(--border);
              border-radius: 6px; cursor: pointer; }}
  .copy-btn:hover {{ color: var(--blue-deep); border-color: var(--blue); }}
  .typing {{ display: none; align-items: center; gap: 5px; padding: 4px 2px 12px 46px; color: var(--dim); font-size: 13px; }}
  .typing .dot {{ width: 7px; height: 7px; background: var(--blue); border-radius: 50%; animation: bounce 1.2s infinite; }}
  .typing .dot:nth-child(2) {{ animation-delay: .2s; }}
  .typing .dot:nth-child(3) {{ animation-delay: .4s; }}
  @keyframes bounce {{ 0%,80%,100% {{ transform: translateY(0); opacity:.4; }} 40% {{ transform: translateY(-5px); opacity:1; }} }}
  #inputbar {{ background: rgba(255,255,255,.92); backdrop-filter: blur(6px);
              border-top: 1px solid var(--border); padding: 12px 20px 10px; }}
  #inputbar .row {{ max-width: 880px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; }}
  #box {{ flex: 1; padding: 10px 14px; border: 1px solid var(--border); border-radius: 12px;
         font-size: 14px; outline: none; resize: none; font-family: inherit; line-height: 1.5;
         background: #fff; max-height: 160px; }}
  #box:focus {{ border-color: var(--blue); box-shadow: 0 0 0 3px rgba(88,166,255,.15); }}
  #send {{ padding: 10px 24px; background: linear-gradient(135deg, var(--blue), var(--blue-deep));
          color: #fff; border: none; border-radius: 12px; font-size: 14px; cursor: pointer;
          box-shadow: 0 2px 6px rgba(47,129,247,.3); transition: transform .1s; }}
  #send:hover {{ transform: translateY(-1px); }}
  #send:disabled {{ opacity: .5; cursor: not-allowed; transform: none; }}
  #stop {{ display: none; padding: 10px 20px; background: #fff; color: #d1242f; border: 1px solid #f0b6bb;
          border-radius: 12px; font-size: 14px; cursor: pointer; }}
  #tools {{ max-width: 880px; margin: 0 auto; display: flex; gap: 8px; align-items: center; }}
  #tips {{ color: var(--dim); font-size: 11px; margin-top: 6px; }}
  .kbd {{ display: inline-block; padding: 0 5px; border: 1px solid var(--border); border-radius: 4px;
         background: #f6f8fa; font-size: 10.5px; color: var(--dim); }}
  #hintbar {{ max-width: 880px; margin: 8px auto 0; display: flex; gap: 14px; flex-wrap: wrap; }}
</style>
</head>
<body>
<header>
  <span class="logo">FORGE</span>
  <span class="slogan">把想法锻造成现实</span>
  <span class="status" id="status">连接中…</span>
</header>
<div id="chat"><div class="day" id="dayline"></div></div>
<div class="typing" id="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span><span>forge 思考中…</span></div>
<div id="inputbar">
  <div class="row">
    <textarea id="box" rows="1" placeholder="问 forge 点什么…（Enter 发送，Shift+Enter 换行，Esc 清空）"></textarea>
    <button id="send">发送</button>
    <button id="stop" title="停止本次生成（Esc）">■ 停止</button>
  </div>
  <div id="hintbar">
    <span id="tips">只读能力可用（检索 / 计算 / 读文件 / 联网）；写操作需回 CLI 执行（Web 端安全默认拒绝）</span>
  </div>
</div>
<script>
const chat = document.getElementById('chat'), box = document.getElementById('box'),
      send = document.getElementById('send'), stopBtn = document.getElementById('stop'),
      typing = document.getElementById('typing'), status = document.getElementById('status');
let busy = false, stopFlag = false, history = [], hIdx = -1;

// ---- 简易 Markdown 渲染（先转义防注入，再还原受控格式）----
function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function renderMd(text) {
  let t = escapeHtml(text);
  // 代码块 ```lang ... ```
  t = t.replace(/```([\\w+-]*)\\n([\\s\\S]*?)```/g, (m, lang, code) =>
    '<pre><code>' + code.replace(/\\n$/, '') + '</code></pre>');
  // 行内代码 `x`
  t = t.replace(/`([^`\\n]+)`/g, '<code>$1</code>');
  // 粗体 **x**
  t = t.replace(/\\*\\*([^*\\n]+)\\*\\*/g, '<strong>$1</strong>');
  // 列表 - item（在 pre 外的行首）
  t = t.split('\\n').map(l => /^[-•]\\s/.test(l.trim()) ? '• ' + l.replace(/^[-•]\\s/, '') : l).join('\\n');
  return t;
}

function now() { return new Date().toLocaleTimeString('zh-CN', {hour:'2-digit', minute:'2-digit'}); }

function addMsg(role, text, animate) {
  const m = document.createElement('div');
  m.className = 'msg ' + (role === 'user' ? 'user' : 'forge');
  const av = document.createElement('div'); av.className = 'avatar'; av.textContent = role === 'user' ? '🧑' : '🔨';
  const body = document.createElement('div'); body.className = 'body';
  const meta = document.createElement('div'); meta.className = 'meta'; meta.textContent = (role === 'user' ? '你' : 'forge') + ' · ' + now();
  const b = document.createElement('div'); b.className = 'bubble';
  if (animate) { b.innerHTML = '<span class="blink">▊</span>'; }
  body.appendChild(meta); body.appendChild(b);
  const copy = document.createElement('button'); copy.className = 'copy-btn'; copy.textContent = '复制';
  copy.onclick = () => { navigator.clipboard.writeText(text); copy.textContent = '✓ 已复制'; setTimeout(()=>copy.textContent='复制',1200); };
  body.appendChild(copy);
  m.appendChild(av); m.appendChild(body);
  chat.appendChild(m); chat.scrollTop = chat.scrollHeight;
  if (animate) typewriter(b, text);
  else b.innerHTML = renderMd(text);
  return b;
}

function typewriter(el, text) {
  let i = 0;
  const raw = renderMd(text);
  // 打字机：先按纯文本逐字，完成后再渲染（避免标签闪烁）
  const plain = escapeHtml(text).split('\\n').join('⏎');
  const step = () => {
    if (stopFlag || i >= plain.length) { el.innerHTML = raw; stopFlag = false; return; }
    const slice = plain.slice(0, i + 1).split('⏎').join('\\n');
    el.textContent = slice + '▊';
    i += 1;
    chat.scrollTop = chat.scrollHeight;
    setTimeout(step, 8);
  };
  step();
}

async function sendMsg() {
  const text = box.value.trim();
  if (!text || busy) return;
  busy = true; stopFlag = false;
  send.disabled = true; stopBtn.style.display = 'inline-block';
  addMsg('user', text); box.value = ''; box.style.height = 'auto';
  typing.style.display = 'flex';
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const d = await r.json();
    if (stopFlag) {
      addMsg('forge', '⏹ 已停止生成。');
    } else if (d.reply) {
      addMsg('forge', d.reply, true);
      if (d.warning) addMsg('forge', '⚠ ' + d.warning);
    } else if (d.error) {
      addMsg('forge', '⚠ ' + d.error);
    }
  } catch (e) {
    addMsg('forge', '⚠ 请求失败：' + e.message);
  }
  typing.style.display = 'none';
  busy = false; send.disabled = false; stopBtn.style.display = 'none';
  box.focus();
}

function stopGen() {
  if (!busy) return;
  stopFlag = true;         // 打字机/等待状态停止
  typing.style.display = 'none';
  busy = false; send.disabled = false; stopBtn.style.display = 'none';
}

async function resetChat() {
  try { await fetch('/api/reset', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'}); }
  catch (e) {}
  chat.innerHTML = '<div class="day" id="dayline">— 已清空对话 —</div>';
  box.focus();
}

async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    status.textContent = '模型：' + (d.model || '—');
  } catch (e) {}
}

send.onclick = sendMsg;
stopBtn.onclick = stopGen;
box.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); return; }
  if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); sendMsg(); return; }
  if (e.key === 'Escape') { if (busy) { stopGen(); } else { box.value = ''; } return; }
  if (e.key === 'ArrowUp' && !box.value && history.length) { hIdx = Math.max(0, (hIdx < 0 ? history.length - 1 : hIdx - 1)); box.value = history[hIdx]; }
  if (e.key === 'ArrowDown' && hIdx >= 0) { hIdx += 1; box.value = hIdx < history.length ? history[hIdx] : ''; }
});
box.addEventListener('input', () => { box.style.height = 'auto'; box.style.height = Math.min(box.scrollHeight, 160) + 'px'; });
box.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    if (box.value.trim()) history.push(box.value.trim());
    hIdx = -1;
  }
});
loadStatus();
document.getElementById('dayline').textContent = new Date().toLocaleDateString('zh-CN', {month:'long', day:'numeric', weekday:'long'});
addMsg('forge', '你好，我是 **forge**——把想法锻造成现实。\\n\\n可以直接问我问题；生成时按 **Esc** 或点「停止」可以中断，输入新问题就是新的引导。');
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    """HTTP 处理：/ 返回页面；/api/chat 对话；/api/reset 重置；/api/status 状态。"""

    server_version = "forge/0.1"

    # ---- 工具方法 ----
    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def log_message(self, fmt, *args):  # 静默默认访问日志（避免刷屏）
        pass

    # ---- 路由 ----
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(_PAGE_HTML.replace("__TITLE__", PAGE_TITLE))
            return
        if self.path == "/api/status":
            self._send_json(self.server.get_status())
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/chat":
            body = self._read_body()
            message = (body.get("message") or "").strip()
            if not message:
                self._send_json({"error": "empty message"}, 400)
                return
            reply, warning = self.server.handle_chat(message)
            out = {"reply": reply}
            if warning:
                out["warning"] = warning
            self._send_json(out)
            return
        if self.path == "/api/reset":
            self.server.handle_reset()
            self._send_json({"ok": True})
            return
        self._send_json({"error": "not found"}, 404)


class ForgeWeb:
    """Web 服务（单会话）：起一个后台线程跑 HTTP 服务，Agent 在事件循环里执行。"""

    def __init__(self, host="127.0.0.1", port=8000, agent=None, auto_open=True):
        self.host = host
        self.port = port
        self.auto_open = auto_open
        self.url = None            # start() 后填入实际地址
        # Web 端写操作安全默认：无交互审批通道 → 全拒绝（只读能力不受影响）
        self.agent = agent or Agent(stream=False, approver=Approver(mode="auto_reject"), show_spinner=False)
        self._loop = None          # Agent 调用所在的事件循环
        self._thread = None        # HTTP 服务线程
        self._server = None
        self._warned = set()       # 已提示过的写操作（避免重复刷提示）

    # ---- 生命周期 ----
    def start(self):
        """起服务（非阻塞，后台线程）。返回实际监听地址。"""
        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._server.handle_chat = self._handle_chat   # 注入回调（Handler 经 server 访问）
        self._server.handle_reset = self._handle_reset
        self._server.get_status = self._status
        self.port = self._server.server_address[1]      # port=0 时取实际端口
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://{self.host}:{self.port}/"
        print(paint(f"  🌐 forge Web 已启动：{self.url}", C.LIGHT_BLUE + C.BOLD))
        print(paint("  （/api/chat 对话 · /api/reset 重置 · Ctrl-C 或 /web stop 停止）", C.DIM))
        if self.auto_open:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass
        return self.url

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        print(paint("  🌐 forge Web 已停止", C.DIM))

    # ---- Agent 执行（在专用事件循环里跑，避免跨线程/跨循环问题） ----
    def _run_async(self, coro):
        """在线程里跑协程：每个请求建一次性事件循环（httpx 连接池按循环隔离，安全）。"""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def _handle_chat(self, message: str):
        """处理一条消息：返回 (reply, warning)。写操作被拒时返回警告提示。"""
        # 长期记忆钩子（与 CLI 一致：先自动沉淀，再召回注入）
        try:
            from .memory import get_memory
            mem = get_memory()
            mem.auto_remember(message)
            ctx = mem.compose_context(message)
            if ctx:
                message = ctx + "\n\n用户问题：" + message
        except Exception:
            pass
        try:
            reply = self._run_async(self.agent.run(message))
        except Exception as e:
            reply = f"⚠ 模型调用失败：{e}"
        # 检测写操作被拒的情况，给一次性提示（不重复刷）
        warning = ""
        if "用户拒绝了该写操作" in reply and "write" not in self._warned:
            self._warned.add("write")
            warning = "提示：写操作（写入/编辑/执行命令）在 Web 端被安全默认拒绝，请回 CLI（forge）执行。"
        return reply, warning

    def _handle_reset(self):
        self.agent.reset()
        self._warned.clear()

    def _status(self):
        """当前角色/模型信息（页面状态栏用）。"""
        try:
            cfg = load_config()
            m = resolve_model(cfg, "default")
            model = m.get("model")
        except Exception:
            model = "?"
        return {"model": model, "host": self.host, "port": self.port}


def start_web(port: int = 8000, auto_open: bool = True) -> ForgeWeb:
    """便捷入口：forge --web [port] 启动 Web 服务。"""
    web = ForgeWeb(port=port, auto_open=auto_open)
    web.start()
    return web
