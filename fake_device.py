"""美容仪模拟器（硬件 Phase 0：不碰真硬件，验证「硬件即工具」全链路）。

模拟一台三合一美容仪（对应真实产品形态）：
  · 电源开关（开/关）
  · 三档强度（1/2/3）
  · 温度传感器（随时间升温，超 45°C 自动断电——模拟过热保护）
  · 电流传感器（读档位对应电流）
  · 运行时长

设计原则（对齐 A24）：
  · 只读操作（读状态/传感器）零风险
  · 写操作（开关/调档）有副作用 → forge 审批层兜底
  · 内置安全保护（超温自断）模拟真实设备的控制平面防线

用法：
  from fake_device import BeautyDevice
  dev = BeautyDevice()
  dev.status()          # 只读
  dev.power_on()        # 写
  dev.set_level(2)      # 写
"""
import random
import threading
import time


class BeautyDevice:
    """模拟美容仪：电源 / 档位 / 温度 / 电流 / 过热保护。"""

    MAX_SAFE_TEMP = 45.0  # 过热保护阈值（°C）
    LEVEL_CURRENT = {0: 0.0, 1: 0.8, 2: 1.6, 3: 2.4}  # 档位 → 电流（A）
    LEVEL_HEAT = {0: 0.0, 1: 0.05, 2: 0.12, 3: 0.20}  # 档位 → 每 100ms 温升（°C）

    def __init__(self, name: str = "三合一美容仪"):
        self.name = name
        self._power = False          # 电源
        self._level = 0              # 档位 0-3
        self._temp = 25.0            # 当前温度（°C，室温起步）
        self._run_seconds = 0.0      # 本次运行时长
        self._safety_tripped = False  # 过热保护是否触发过
        self._lock = threading.Lock()
        self._timer = None
        self._started_at = None

    # ---------- 内部：温度模拟（开机后每 100ms 升温，模拟真实发热） ----------
    def _tick(self):
        with self._lock:
            if self._power:
                self._temp += self.LEVEL_HEAT[self._level] + random.uniform(-0.01, 0.01)
                self._run_seconds = time.time() - self._started_at
                # 过热保护：超阈值自动断电（模拟真实设备的硬件安全线）
                if self._temp >= self.MAX_SAFE_TEMP:
                    self._power = False
                    self._level = 0
                    self._safety_tripped = True
                    if self._timer:
                        self._timer.cancel()
                        self._timer = None
                    return
        if self._power:
            self._timer = threading.Timer(0.1, self._tick)
            self._timer.daemon = True
            self._timer.start()

    # ---------- 只读操作（无副作用，Agent 可随时调） ----------
    def status(self) -> dict:
        """读完整状态（只读，零风险）。"""
        with self._lock:
            return {
                "device": self.name,
                "power": self._power,
                "level": self._level,
                "temperature_c": round(self._temp, 1),
                "current_a": self.LEVEL_CURRENT[self._level] if self._power else 0.0,
                "run_seconds": round(self._run_seconds, 1),
                "safety_tripped": self._safety_tripped,
            }

    # ---------- 写操作（有副作用，需审批层兜底） ----------
    def power_on(self) -> dict:
        """开机（写操作）。"""
        with self._lock:
            if self._safety_tripped:
                return {"ok": False, "reason": "过热保护已触发，需手动复位后才可开机"}
            self._power = True
            self._level = 1
            self._started_at = time.time()
            self._run_seconds = 0.0
        self._timer = threading.Timer(0.1, self._tick)
        self._timer.daemon = True
        self._timer.start()
        return {"ok": True, "status": self.status()}

    def power_off(self) -> dict:
        """关机（写操作）。"""
        with self._lock:
            self._power = False
            self._level = 0
            if self._timer:
                self._timer.cancel()
                self._timer = None
        return {"ok": True, "status": self.status()}

    def set_level(self, level: int) -> dict:
        """调档（写操作）。"""
        if level not in (1, 2, 3):
            return {"ok": False, "reason": f"档位只能是 1/2/3，收到 {level!r}"}
        with self._lock:
            if not self._power:
                return {"ok": False, "reason": "设备未开机，先开机再调档"}
            if self._safety_tripped:
                return {"ok": False, "reason": "过热保护已触发，需复位"}
            self._level = level
        return {"ok": True, "status": self.status()}

    def reset_safety(self) -> dict:
        """复位过热保护（写操作，模拟维修人员介入）。"""
        with self._lock:
            self._safety_tripped = False
            self._temp = 25.0
            self._level = 0
            self._power = False
        return {"ok": True, "reason": "安全复位完成", "status": self.status()}
