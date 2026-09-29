"""M3 #4 审批层：写操作执行前的人工/可编程确认（对应 A06 只读分级的审批补全）。

三种模式：
  interactive   CLI 交互询问（默认，forge 主对话用）
  callback      approver 回调函数 approve(name, summary) -> bool（Web/自动化/测试用）
  auto_approve  全部放行（无人值守批量/并行任务用）
  auto_reject   全部拒绝（演练/策略控制用）

设计要点：
  - 只拦 is_write=True 的工具（write_file/edit_file/run_command 等），只读工具零打扰
  - 拒绝返回 False，调用方把「用户拒绝」作为工具结果喂回模型，让它调整方案
  - 与 A06 只读分级协同：分级是静态判定，审批是执行前的人机闸门
"""
from .console import C, paint


class Approver:
    def __init__(self, mode="interactive", callback=None, auto_approve=False):
        self.mode = mode
        self.callback = callback  # (name, summary) -> bool
        self.auto_approve = auto_approve
        self.decisions = []  # [(name, summary, bool)] 审计记录

    def approve(self, name: str, summary: str) -> bool:
        """询问是否允许执行写操作；返回 True 放行 / False 拒绝。"""
        if self.auto_approve:
            self.decisions.append((name, summary, True))
            return True
        if self.mode == "auto_reject":
            self.decisions.append((name, summary, False))
            return False
        if self.callback is not None:
            ok = bool(self.callback(name, summary))
            self.decisions.append((name, summary, ok))
            return ok
        # 交互模式：红字警告 + y/N
        while True:
            print(paint(f"  ⚠ 写操作请求：{name}({summary[:100]})", C.SKY))
            v = input(paint("  允许执行吗？[y/N] ", C.RED)).strip().lower()
            if v in ("y", "yes"):
                self.decisions.append((name, summary, True))
                return True
            if v in ("", "n", "no"):
                self.decisions.append((name, summary, False))
                return False
            print(paint("  请输入 y / n", C.DIM))
