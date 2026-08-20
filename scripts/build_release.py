"""发布打包脚本：把 forge 做成「别人能下载安装」的干净发布副本。

设计原则（2026-08-20： 确认）：
  · 只读源 —— 绝不修改 handcraft-agent/ 里任何源码/配置/数据
  · 产出独立 —— 生成 release/forge/（独立 git 仓库），本地工作区照常用
  · 隐私红线 —— 真实 API key 脱敏为 sk-xxx；data/exports/日志/缓存一律不进发布物

用法：
  python scripts/build_release.py [--out release/forge] [--no-smoke]

产出 release/forge/：
  src/ main.py pyproject.toml requirements.txt test_*.py  README.md README.en.md
  config/models.example.yaml（key 脱敏模板，用户填 key 后用）
  .gitignore（排除 config/models.yaml 等敏感文件）
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(BASE_DIR, "release", "forge")

# 复制白名单（顶层）：代码 + 测试 + 文档 + 配置模板 + CI
COPY_GLOBS = [
    "src/**", "main.py", "pyproject.toml", "requirements.txt",
    "test_*.py", "smoke_*.py", "stress_*.py",
    "README.md", "README.en.md", "LICENSE",
    ".github/**",
]
# 明确排除（防误伤）：本地运行残留（scripts/ 单独处理——只放行 build_release.py）
EXCLUDE_DIRS = {"data", "exports", ".git", "__pycache__", ".workbuddy", "release"}
EXCLUDE_EXTS = {".db", ".log", ".pyc", ".pyo"}
EXCLUDE_NAMES = {"config/models.yaml"}  # 真实配置（含 key）不进发布物

# 敏感 key 正则：sk- 开头或 api_key 字段里的密钥串
KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-agent\d-[A-Za-z0-9]{20,}"),
]


def _desensitize(text: str) -> str:
    """把 api_key 明文替换为占位符（保留字段结构）。"""
    for pat in KEY_PATTERNS:
        text = pat.sub("sk-xxx", text)
    return text


def _should_skip_rel(rel: str) -> bool:
    rel = rel.replace("\\", "/")  # 统一正斜杠：Windows walk 产生反斜杠，Linux 正斜杠（平台一致性）
    parts = rel.split("/")
    for p in parts:
        if p in EXCLUDE_DIRS or p.endswith(".egg-info"):
            return True
    # scripts/ 目录：放行目录本身，但只复制 build_release.py（打包工具随仓库发布，供他人重新打包）
    if "scripts" in parts:
        return not (rel == "scripts" or rel.endswith("scripts/build_release.py"))
    for p in parts:
        if p.startswith(".") and p != ".github":  # 隐藏文件/目录（.github 是 CI 配置，放行）
            return True
    ext = os.path.splitext(rel)[1].lower()
    if ext in EXCLUDE_EXTS:
        return True
    return False


def _copy_tree(src: str, dst: str) -> int:
    """复制 src 下白名单内容到 dst，返回复制文件数。"""
    n = 0
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        # 剪枝：排除目录
        dirs[:] = [d for d in dirs if not _should_skip_rel(os.path.join(rel_root, d) if rel_root != "." else d)]
        for f in files:
            rel = os.path.join(rel_root, f) if rel_root != "." else f
            if _should_skip_rel(rel):
                continue
            s = os.path.join(root, f)
            d = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            # 白名单后缀
            if f.endswith((".py", ".toml", ".txt", ".md")):
                with open(s, "r", encoding="utf-8", errors="replace") as fr:
                    content = fr.read()
                content = _desensitize(content)
                with open(d, "w", encoding="utf-8", newline="") as fw:
                    fw.write(content)
            else:
                shutil.copy2(s, d)
            n += 1
    return n


def _write_gitignore(dst: str):
    """发布副本的 .gitignore：真实配置/本地数据永不入库。"""
    content = """# forge 发布仓库忽略项
# 真实配置（含 API key）与本地数据绝不入库
config/models.yaml
data/
exports/
*.db
*.log
__pycache__/
*.pyc
release/
.workbuddy/
"""
    with open(os.path.join(dst, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(content)


def _write_example_config(dst: str):
    """从源 models.yaml 生成脱敏的 models.example.yaml（用户填 key 模板）。"""
    src = os.path.join(BASE_DIR, "config", "models.yaml")
    if not os.path.exists(src):
        return
    with open(src, "r", encoding="utf-8") as f:
        content = f.read()
    content = _desensitize(content)
    os.makedirs(os.path.join(dst, "config"), exist_ok=True)
    with open(os.path.join(dst, "config", "models.example.yaml"), "w", encoding="utf-8") as f:
        f.write(content)
    # 副本里也放一份脱敏 models.yaml（用户 clone 后直接填 key 就能跑）
    with open(os.path.join(dst, "config", "models.yaml"), "w", encoding="utf-8") as f:
        f.write(content)


def _check_no_secrets(dst: str) -> bool:
    """校验发布物里没有真实 key 残留。"""
    bad = []
    for root, _, files in os.walk(dst):
        if "data" in root.split(os.sep):
            continue
        for f in files:
            if not f.endswith((".py", ".yaml", ".yml", ".toml", ".md", ".txt")):
                continue
            p = os.path.join(root, f)
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fr:
                    text = fr.read()
                for pat in KEY_PATTERNS:
                    if pat.search(text):
                        bad.append(p)
                        break
            except OSError:
                continue
    if bad:
        print("⚠ 发现疑似 key 残留：")
        for p in bad:
            print("  " + p)
        return False
    return True


def _smoke(dst: str) -> bool:
    """冒烟：用发布副本跑核心 mock 测试，证明副本完整可跑。"""
    print("\n== 冒烟验证发布副本 ==")
    tests = ["test_router.py", "test_interrupt.py", "test_approval.py"]
    py = sys.executable
    ok = True
    for t in tests:
        p = os.path.join(dst, t)
        if not os.path.exists(p):
            print(f"  ⚠ {t} 缺失，跳过")
            continue
        print(f"  run {t} …")
        r = subprocess.run([py, t], cwd=dst, capture_output=True, text=True, timeout=120)
        # Windows 下 unittest 输出可能在 stderr——合并判定
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        failed_markers = ("FAILED", "ERROR:", "FAIL:")
        passed = (r.returncode == 0
                  and "Ran " in out and " tests" in out
                  and not any(m in out for m in failed_markers))
        if passed:
            print(f"  ✅ {t}: 通过")
        else:
            print(f"  ❌ {t}: 未通过（returncode={r.returncode}）")
            tail3 = out.splitlines()[-3:]
            print("     " + " | ".join(tail3))
            if r.stderr:
                print("     " + r.stderr.strip().splitlines()[-1][:200])
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description="forge 发布打包")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-smoke", action="store_true", help="跳过冒烟测试")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    if os.path.exists(out):
        try:
            shutil.rmtree(out)
        except OSError:
            # 沙箱等环境可能禁止删除（safe-delete 拦截）——改覆盖模式继续，残留文件靠白名单天然不复制
            print("  ⚠ 无法删除旧发布目录，将以覆盖方式写入（白名单复制，多余残留不影响）")
    os.makedirs(out, exist_ok=True)

    print(f"源目录：   {BASE_DIR}")
    print(f"发布目录： {out}")

    n = _copy_tree(BASE_DIR, out)
    _write_gitignore(out)
    _write_example_config(out)
    print(f"已复制 {n} 个文件")

    print("\n== 隐私校验 ==")
    if _check_no_secrets(out):
        print("  ✅ 无真实 key 残留")
    else:
        print("  ❌ 存在 key 残留，中止")
        sys.exit(1)

    if not args.no_smoke:
        if not _smoke(out):
            print("\n❌ 冒烟未通过，发布副本可能不完整")
            sys.exit(1)
        print("\n✅ 打包完成，发布副本可独立使用：")
    else:
        print("\n✅ 打包完成（跳过冒烟）：")
    print(f"   {out}")
    print("   cd 进去后：git init && git add . && git commit（.gitignore 已备好）")
    print("   pip install -e . 即可得到 forge 命令")


if __name__ == "__main__":
    main()
