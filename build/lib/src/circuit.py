"""熔断器（Circuit Breaker）：模型通道级故障隔离（对应 A07 错误处理·#5 熔断）。

为什么需要它：
  llm.chat / stream_chat 已有「主角色 → fallback 角色」降级链，但**没有跨调用的状态**。
  某端点持续故障时，每次调用仍会傻傻重试 3 次（指数退避）再降级，延迟和资源白烧，
  还可能在高并发下对故障端点造成雪崩式打满。

熔断器做的事：
  按「角色」维度维护一个三态机：
    CLOSED    正常放行；记录连续失败数，达到阈值 → OPEN
    OPEN      熔断中；冷却期内直接拒绝（快速失败），不浪费重试；冷却到期 → HALF_OPEN
    HALF_OPEN 放行一次探测；成功 → CLOSED 恢复，失败 → 回到 OPEN
  集成进 llm 后：故障角色被熔断时，chat 直接跳过它去试下一个角色，把降级链从
  「重试 3 次再降级」升级成「连续失败 N 次后瞬间跳过」，延迟与下游压力骤降。

设计要点：
  - 这是「模型通道」维度的熔断。工具级禁用（A07 字面定义）是同一 breaker 的扩展点，
    后续可在 tools 执行层用同名 registry 按工具名挂 breaker，本次不强制接入。
  - 阈值 / 冷却 / 半开探测次数全部配置驱动（models.yaml 的 circuit_breaker 段）。
  - 时间用可注入的 clock（默认 time.time），测试里可自由拨钟。
"""
from typing import Callable, Dict, Optional


class CircuitOpenError(Exception):
    """熔断器处于 OPEN（熔断中）时快速失败抛出的异常。

    message 已带角色名与冷却到期时间，便于上层（agent / CLI）直接展示。
    """
    pass


class CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown: float = 30.0,
        half_open_max: int = 1,
        clock: Callable[[], float] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.half_open_max = half_open_max
        self._clock = clock or _default_clock
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._half_open_trials = 0

    # ---- 状态查询 ----
    @property
    def is_open(self) -> bool:
        """当前是否处于「熔断拒绝」态（OPEN 且冷却未到期）。"""
        return self.state == self.OPEN and self._clock() < self.opened_at + self.cooldown

    @property
    def cooldown_remaining(self) -> float:
        if self.state != self.OPEN:
            return 0.0
        return max(0.0, (self.opened_at + self.cooldown) - self._clock())

    def allow(self) -> bool:
        """是否允许发起一次调用（纯状态判断，不阻塞）。

        返回 False 表示当前应被熔断拦截（调用方应跳过该角色 / 走降级）。
        冷却到期时会把 OPEN 自动推进到 HALF_OPEN 并放行一次探测。
        """
        if self.state == self.OPEN:
            if self._clock() >= self.opened_at + self.cooldown:
                self.state = self.HALF_OPEN
                self._half_open_trials = 0
                return True
            return False
        return True

    # ---- 结果回报 ----
    def record_success(self) -> None:
        """调用成功：HALF_OPEN 探测成功 → 恢复 CLOSED；否则清零连续失败。"""
        if self.state == self.HALF_OPEN:
            self.state = self.CLOSED
        self.failures = 0
        self._half_open_trials = 0

    def record_failure(self) -> None:
        """调用失败：HALF_OPEN 探测失败 → 立即重开；CLOSED 累计失败达阈值 → 开。"""
        if self.state == self.HALF_OPEN:
            self._open()
            return
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self._open()

    def _open(self) -> None:
        self.state = self.OPEN
        self.opened_at = self._clock()
        self.failures = 0
        self._half_open_trials = 0

    def reset(self) -> None:
        """手动复位（CLI / 配置热重载用）。"""
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._half_open_trials = 0

    def snapshot(self) -> dict:
        """可序列化的状态快照（CLI / 调试用）。"""
        return {
            "name": self.name,
            "state": self.state,
            "failures": self.failures,
            "cooldown_remaining": round(self.cooldown_remaining, 1),
        }


def _default_clock() -> float:
    import time
    return time.time()


class CircuitRegistry:
    """按 key 维度管理一组熔断器（默认 key = 角色名）。全局单例。"""

    def __init__(self, defaults: Optional[dict] = None):
        self._breakers: Dict[str, CircuitBreaker] = {}
        d = defaults or {}
        self.defaults = {
            "failure_threshold": d.get("failure_threshold", 3),
            "cooldown": d.get("cooldown", 30.0),
            "half_open_max": d.get("half_open_max", 1),
        }

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, **self.defaults)
        return self._breakers[name]

    def reset(self, name: Optional[str] = None) -> None:
        if name:
            if name in self._breakers:
                self._breakers[name].reset()
        else:
            for b in self._breakers.values():
                b.reset()

    def snapshot(self) -> Dict[str, dict]:
        return {n: b.snapshot() for n, b in self._breakers.items()}


_REGISTRY: Optional[CircuitRegistry] = None


def get_circuit_registry(cfg: Optional[dict] = None) -> CircuitRegistry:
    """懒初始化全局熔断器注册表（配置驱动默认参数）。

    cfg 为 models.yaml 整体（含 circuit_breaker 段）。首次调用用 cfg 定默认值；
    之后调用即便传不同 cfg 也不改变已建实例的默认值（避免运行期抖动）。
    需要按新配置重建时显式调用 reset_circuit_registry() 或 breaker.reset()。
    """
    global _REGISTRY
    if _REGISTRY is None:
        cb = (cfg or {}).get("circuit_breaker", {}) or {}
        _REGISTRY = CircuitRegistry(
            {
                "failure_threshold": cb.get("failure_threshold", 3),
                "cooldown": cb.get("cooldown", 30.0),
                "half_open_max": cb.get("half_open_max", 1),
            }
        )
    return _REGISTRY


def reset_circuit_registry() -> None:
    """清空全局注册表（配置热重载 / 测试隔离用）。"""
    global _REGISTRY
    _REGISTRY = None
