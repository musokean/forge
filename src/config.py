"""配置加载：读 config/models.yaml，按角色解析模型（对应 A17 多模型路由 + 模型配置层）。"""
import os

import yaml

# 项目根目录 = 本文件上一级（src/）的上一级；路径基于代码位置，与启动时的 cwd 无关
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 内置最小配置模板：pip 安装（无 config/ 目录）时自动生成，避免 FileNotFoundError（2026-08-24 实测）
_DEFAULT_CONFIG_TEMPLATE = """\
# forge 默认配置（首次运行自动生成）
# 修改这个文件即可调整 forge 的模型 / 角色，保存即生效。
models:
  deepseek_v4_flash:
    label: DeepSeek V4 Flash
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-v4-flash
  deepseek_v4_pro:
    label: DeepSeek V4 Pro
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-v4-pro
  qwen_cloud_plus:
    label: 通义千问 Plus
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key_env: DASHSCOPE_API_KEY
    model: qwen-plus
roles:
  default:   { model: deepseek_v4_flash, label: 主力, purpose: 常规对话主力 }
router:
  role: default
debate:
  rounds: 2
  roles: []
circuit_breaker:
  failure_threshold: 3
  cooldown: 30
  half_open_max: 1
reflect:
  enabled: false
  min_score: 6
  max_rounds: 1
  judge_role: fallback
"""


def load_config(path=None):
    if path is None:
        path = os.path.join(_BASE_DIR, "config", "models.yaml")
    elif not os.path.isabs(path):
        path = os.path.join(_BASE_DIR, path)  # 相对路径也基于项目根解析
    if not os.path.exists(path):
        _auto_generate_config(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _auto_generate_config(path):
    """配置文件不存在时（如 pip 安装的 wheel 不含 config/），自动生成默认配置。

    真实用户从 GitHub pip install 后首次运行不再崩 FileNotFoundError（2026-08-24 实测）。
    key 走环境变量（api_key_env），用户设好环境变量或手动改文件即可。
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(_DEFAULT_CONFIG_TEMPLATE)
        print(f"⚙ 未找到配置文件，已自动生成默认配置：{path}")
        print("   · 设好环境变量（如 DEEPSEEK_API_KEY）即可使用，或直接编辑该文件填 key")
    except Exception as e:
        print(f"⚠ 自动生成配置失败（{e}），返回空配置；请手动创建 {path}")


def _models_list(cfg):
    """把 models 段统一成 [(name, model_dict)]，兼容两种写法：
       models:                      models:
         - name: local_qwen           local_qwen:
             base_url: ...              base_url: ...
                                         （key 即别名，少写一个 name 字段）
    """
    ms = cfg.get("models", [])
    if isinstance(ms, dict):
        return list(ms.items())
    return [(m.get("name"), m) for m in ms]


def _with_api_key(name, m: dict, role: str) -> dict:
    """复制模型 dict，补上 name，并解析出可用的 api_key（字面 api_key 优先，否则取环境变量）。"""
    out = dict(m)
    out["name"] = name
    key_env = m.get("api_key_env")
    out["api_key"] = m.get("api_key") or (os.environ.get(key_env) if key_env else None)
    out["role"] = role
    return out


def resolve_model(cfg, role="default"):
    """按「角色名」或「模型别名」解析出模型配置（返回带元信息的 dict）。

    返回值除模型本身字段（name/base_url/api_key/model…）外，还附带：
      - label / purpose：仅当 role 是 roles 里的角色时，才有（中文显示名与用途）
      - api_key        ：已解析好的可用 key（字面优先，否则从环境变量取）
    兼容两种 roles 写法：
      roles:
        default: qwen3_local                                          # 简写：别名
        reasoning: { model: qwen3_local, label: 推理, purpose: ... }  # 完整：带元信息
    """
    models = _models_list(cfg)
    roles = cfg.get("roles", {})
    entry = roles.get(role)
    if entry is None:
        # 不是角色名 → 直接当 models 里的别名找（保留按模型名直调的能力）
        for name, m in models:
            if name == role:
                return _with_api_key(name, m, role)
        raise ValueError(f"角色/别名 {role} 未在 roles 或 models 里定义")

    if isinstance(entry, str):
        alias, label, purpose = entry, None, None
    else:
        alias = entry.get("model")
        label = entry.get("label")
        purpose = entry.get("purpose")

    for name, m in models:
        if name == alias:
            out = _with_api_key(name, m, role)
            if label:
                out["label"] = label
            if purpose:
                out["purpose"] = purpose
            return out
    raise ValueError(f"角色 {role} 指向的模型别名 {alias} 在 models 里找不到")
