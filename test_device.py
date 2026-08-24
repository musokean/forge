"""硬件 Phase 0 模拟器测试（A24 硬件即工具）：模拟器行为 / 工具注册 / 只读分级 / 审批拦截。

运行：pytest test_device.py
"""
import json
import sys
import time
import unittest

sys.path.insert(0, ".")

from fake_device import BeautyDevice
from src.tools import TOOLS, device_level, device_power, device_status, execute, is_write
from src.approval import Approver


class TestBeautyDevice(unittest.TestCase):
    """模拟器行为。"""

    def test_initial_state(self):
        dev = BeautyDevice()
        st = dev.status()
        self.assertFalse(st["power"])
        self.assertEqual(st["level"], 0)
        self.assertGreaterEqual(st["temperature_c"], 25.0)

    def test_power_on_sets_level_1_and_heats(self):
        dev = BeautyDevice()
        r = dev.power_on()
        self.assertTrue(r["ok"])
        st = dev.status()
        self.assertTrue(st["power"])
        self.assertEqual(st["level"], 1)
        time.sleep(0.35)  # 等 3 个 tick（~0.1s each），温度应上升
        st2 = dev.status()
        self.assertGreater(st2["temperature_c"], st["temperature_c"])
        dev.power_off()

    def test_set_level_requires_power(self):
        dev = BeautyDevice()
        r = dev.set_level(2)
        self.assertFalse(r["ok"])  # 未开机拒绝
        self.assertIn("未开机", r["reason"])

    def test_set_level_valid_after_power(self):
        dev = BeautyDevice()
        dev.power_on()
        r = dev.set_level(3)
        self.assertTrue(r["ok"])
        self.assertEqual(dev.status()["level"], 3)
        self.assertAlmostEqual(dev.status()["current_a"], 2.4, places=1)
        dev.power_off()

    def test_set_level_invalid_rejected(self):
        dev = BeautyDevice()
        dev.power_on()
        r = dev.set_level(9)
        self.assertFalse(r["ok"])
        self.assertEqual(dev.status()["level"], 1)  # 档位没变
        dev.power_off()

    def test_overheat_safety_trip(self):
        """过热保护：温度超阈值自动断电（控制平面防线）。"""
        dev = BeautyDevice()
        dev.power_on()
        dev.set_level(3)
        # 模拟跑很久（直接灌温度到阈值附近，不用真等）
        dev._temp = dev.MAX_SAFE_TEMP - 0.5
        deadline = time.time() + 3.0
        while time.time() < deadline and not dev.status()["safety_tripped"]:
            time.sleep(0.15)
        st = dev.status()
        self.assertTrue(st["safety_tripped"], "应触发过热保护")
        self.assertFalse(st["power"], "过热应自动断电")
        # 复位后可再开机
        r = dev.reset_safety()
        self.assertTrue(r["ok"])
        self.assertTrue(dev.power_on()["ok"])

    def test_power_off_stops_heating(self):
        dev = BeautyDevice()
        dev.power_on()
        time.sleep(0.25)
        dev.power_off()
        t1 = dev.status()["temperature_c"]
        time.sleep(0.3)
        t2 = dev.status()["temperature_c"]
        self.assertAlmostEqual(t1, t2, delta=0.3)  # 关机后温度不再显著上升


class TestHardwareTools(unittest.TestCase):
    """工具注册 / 只读分级 / 审批拦截。"""

    def test_tools_registered(self):
        self.assertIn("device_status", TOOLS)
        self.assertIn("device_power", TOOLS)
        self.assertIn("device_level", TOOLS)

    def test_read_write_classification(self):
        self.assertFalse(is_write("device_status"))  # 只读
        self.assertTrue(is_write("device_power"))  # 写
        self.assertTrue(is_write("device_level"))  # 写

    def test_device_status_is_ready_only_exec(self):
        """只读工具直接调（零审批）。"""
        r = json.loads(device_status())
        self.assertIn("device", r)
        self.assertIn("power", r)

    def test_approval_rejects_write_keeps_device_off(self):
        """审批拒绝 → 设备不动（写操作被闸住）。"""
        approver = Approver(mode="auto_reject")
        self.assertTrue(is_write("device_power"))
        # 模拟 Agent 审批闸：写操作先过审批
        decision = approver.approve("device_power", "开机")
        self.assertFalse(decision, "auto_reject 应拒绝")
        # 设备仍是关机（没被执行）
        st = json.loads(device_status())
        self.assertFalse(st["power"])

    def test_approval_allows_write_executes(self):
        """审批放行 → 设备开机成功（端到端）。"""
        approver = Approver(auto_approve=True)
        decision = approver.approve("device_power", "开机")
        self.assertTrue(decision)
        r = json.loads(device_power("on"))
        self.assertTrue(r["ok"])
        st = json.loads(device_status())
        self.assertTrue(st["power"])
        device_power("off")  # 清理

    def test_device_level_end_to_end(self):
        r1 = json.loads(device_power("on"))
        self.assertTrue(r1["ok"])
        r2 = json.loads(device_level(2))
        self.assertTrue(r2["ok"])
        self.assertEqual(r2["status"]["level"], 2)
        device_power("off")

    def test_invalid_action_rejected(self):
        r = json.loads(device_power("explode"))
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
