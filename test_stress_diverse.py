"""M1 地狱压测变体 · 多任务混跑：N 种不同任务混跑 50 轮。

与原 test_stress_hell.py 的 long_haul（重复同一简单任务）不同，本脚本验证的是
「**不同任务路径**下不崩、不卡死循环、答案有效、副作用生效」——
覆盖 calc / list / search / time / read / write / edit / command 八条本地工具路径，
可选加上 web_search / web_fetch 两条联网路径（--net）。

每轮：新建 Agent（干净上下文）→ 从任务池挑一类 → 跑 → 校验返回非空 + 副作用成立。
前 len(pool) 轮确定性覆盖每类一次，之后随机混跑，确保 50 轮里每类都被反复打。
"""
import argparse
import asyncio
import os
import random
import sys
import time

sys.path.insert(0, ".")

from src.agent import Agent

TEMP = os.environ.get("TEMP", "/tmp")
FIX = lambda n: os.path.join(TEMP, f"forge_diverse_{n}.txt")
WRITE = lambda n: os.path.join(TEMP, f"forge_write_{n}.txt")


def build_tasks(n, net):
    """构建第 n 轮可用的任务池。每项 = (label, prompt, check_fn)。"""
    f = FIX(n)
    tasks = [
        ("calc",
         "请计算 (123 * 456 + 789) / 3 的最终结果，只给出数字",
         lambda a, n: any(ch.isdigit() for ch in a)),
        ("list",
         "列出当前工作目录下的文件和文件夹",
         lambda a, n: len(a.strip()) > 0),
        ("search",
         f"在目录 {os.getcwd()} 中搜索文件名包含 'main' 的文件并列出",
         lambda a, n: len(a.strip()) > 0),
        ("time",
         "现在是什么时间？用一句话回答",
         lambda a, n: len(a.strip()) > 0),
        ("read",
         f"请读取文件 {f} 的内容，用一句话概括它",
         lambda a, n: len(a.strip()) > 0),
        ("write",
         f"请创建文件 {WRITE(n)} 并写入一行：hello round {n}",
         lambda a, n: os.path.exists(WRITE(n))),
        ("edit",
         f"请编辑文件 {f}，把里面的 'BASE' 改成 'EDITED round {n}'",
         lambda a, n: "EDITED" in open(f, encoding="utf-8", errors="ignore").read()),
        ("cmd",
         f"请执行命令 echo forge-stress-{n} 并返回输出",
         lambda a, n: f"forge-stress-{n}" in a),
    ]
    if net:
        tasks += [
            ("wsearch",
             "搜索「2026 年最新开源大语言模型」的两条要点",
             lambda a, n: len(a.strip()) > 0),
            ("wfetch",
             "抓取 https://example.com 的网页标题或主要内容，简要说明",
             lambda a, n: len(a.strip()) > 0),
        ]
    return tasks


async def diverse_run(rounds=50, net=False, seed=42):
    random.seed(seed)
    pool = [t[0] for t in build_tasks(0, net)]
    seen = {p: 0 for p in pool}
    crashes = 0
    fails = 0
    t0 = time.time()

    for i in range(1, rounds + 1):
        # 预置本轮 fixture（read/edit 任务依赖已知内容）
        with open(FIX(i), "w", encoding="utf-8") as fh:
            fh.write(f"ROUND {i} BASE\n")

        tasks = build_tasks(i, net)
        # 前 len(tasks) 轮确定性覆盖每类一次，之后随机混跑
        pick = tasks[(i - 1) % len(tasks)] if i <= len(tasks) else random.choice(tasks)
        label, prompt, check = pick
        seen[label] = seen.get(label, 0) + 1

        a = Agent(max_context_tokens=4000, max_steps=8, stream=False, name=label)
        try:
            ans = await a.run(prompt)
        except Exception as e:
            crashes += 1
            print(f"  ❌ 第{i}轮[{label}] 崩溃: {type(e).__name__}: {e}")
            continue

        ok = bool(ans and ans.strip())
        if ok and check:
            try:
                ok = bool(check(ans, i))
            except Exception as e:
                ok = False
                print(f"  ⚠️ 第{i}轮[{label}] 校验抛异常: {e}")
        if not ok:
            fails += 1
            print(f"  ❌ 第{i}轮[{label}] 答案无效/校验失败: {(ans or '')[:60]}")
        elif i % 10 == 0:
            print(f"  …第{i}轮 [{label}] ✅ 累计崩{crashes}/无效{fails}")

    # 清理临时文件
    for i in range(1, rounds + 1):
        for p in (FIX(i), WRITE(i)):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    dt = time.time() - t0
    print(f"\n=== 多任务混跑 {rounds} 轮完成：崩 {crashes} · 答案无效 {fails} · "
          f"任务覆盖 {seen} · 耗时 {dt:.0f}s ===")
    return crashes, fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="多任务混跑压测变体")
    ap.add_argument("--rounds", type=int, default=50, help="混跑轮数（默认 50）")
    ap.add_argument("--net", action="store_true", help="包含真实联网任务 web_search/web_fetch")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    c, f = asyncio.run(diverse_run(rounds=args.rounds, net=args.net, seed=args.seed))
    sys.exit(1 if (c or f) else 0)
