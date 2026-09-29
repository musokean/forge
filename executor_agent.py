"""客户端执行器独立入口（#17：装在被控 PC 上跑，只出站，不开放任何入站端口）。

    python executor_agent.py --center http://127.0.0.1:8080 --token <APIKEY> --id pc-01
    python executor_agent.py --center http://127.0.0.1:8080 --token <APIKEY> --once     # 自检一次就退
    python executor_agent.py --list-caps                                                # 看本机能声明哪些能力

默认值取自配置 `executor.client` 段（jail 目录 / 能力白名单 / 阶段 / 体积上限），命令行参数可覆盖。
GUI 能力（screenshot / input）需要可选依赖：pip install "handcraft-agent[executor]"
——没装就不声明这两个能力（**不假装有**）。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.executor import ExecutorClient, default_driver          # noqa: E402


def load_client_conf():
    try:
        from src.config import load_config

        return ((load_config() or {}).get("executor") or {}).get("client") or {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description="forge 客户端执行器（#17：中心大脑 + 分布式手脚）")
    ap.add_argument("--center", default=os.environ.get("FORGE_CENTER", "http://127.0.0.1:8080"),
                    help="中心地址（forge --serve 起的服务）")
    ap.add_argument("--token", default=os.environ.get("FORGE_EXECUTOR_TOKEN", ""),
                    help="中心 API key（也可用环境变量 FORGE_EXECUTOR_TOKEN）")
    ap.add_argument("--id", dest="device_id", default="", help="设备号（默认 主机名-PID）")
    ap.add_argument("--root", default="", help="路径 jail 根目录（默认执行器启动目录）")
    ap.add_argument("--allow", default="", help="能力白名单（逗号分隔，如 shell,read_file,screenshot）")
    ap.add_argument("--stage", default="", help="本机阶段 readonly / low_risk / approval / closed_loop")
    ap.add_argument("--max-bytes", type=int, default=0, help="单次读写/截图体积上限")
    ap.add_argument("--once", action="store_true", help="注册一次并只轮询一轮就退出（自检用）")
    ap.add_argument("--duration", type=float, default=0, help="跑多少秒后退出（0=一直跑）")
    ap.add_argument("--idle-timeout", type=float, default=0, help="多久没命令就退出（0=不退出）")
    ap.add_argument("--list-caps", action="store_true", help="只打印本机可声明的能力后退出")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    conf = load_client_conf()
    if args.root:
        conf["root"] = args.root
    if args.allow:
        conf["allow_caps"] = [c.strip() for c in args.allow.split(",") if c.strip()]
    if args.stage:
        conf["stage"] = args.stage
    if args.max_bytes:
        conf["max_bytes"] = args.max_bytes

    client = ExecutorClient(center=args.center, token=args.token, device_id=args.device_id,
                            client_conf=conf, verbose=not args.quiet)

    if args.list_caps:
        ok, why = client.driver.probe()
        print(f"  驱动：{client.driver.kind}（{'可用' if ok else '不可用：' + why}）")
        print(f"  可声明能力：{client.capabilities()}")
        print(f"  路径 jail：{client.policy.root}")
        print(f"  本机阶段：{client.policy.stage} · 体积上限：{client.policy.max_bytes} 字节")
        return 0

    print(f"  [exec] 连接中心 {args.center} · 设备 {client.device_id} · 目录 jail {client.policy.root}",
          flush=True)
    try:
        reg = client.register()
    except Exception as e:
        print(f"  [exec] ❌ 注册失败：{e}", flush=True)
        print("         中心要先起来：forge --serve（或 python main.py --serve）", flush=True)
        return 1
    print(f"  [exec] ✅ 已注册 · 能力 {reg.get('device', {}).get('capabilities')} · "
          f"中心阶段 {reg.get('hub_stage')} · 长轮询 {reg.get('poll_wait')}s", flush=True)

    if args.once:
        out = client.loop(duration=0.2)          # 轮询一轮（0.2s）即退，用于连通性自检
        print(f"  [exec] 自检结束：{out}", flush=True)
        return 0

    try:
        stats = client.loop(duration=args.duration or None,
                            idle_timeout=args.idle_timeout or None)
    except KeyboardInterrupt:
        stats = dict(client.stats)
    print(f"  [exec] 退出 · 统计 {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
