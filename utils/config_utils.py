from omegaconf import OmegaConf,DictConfig
import os 
from pathlib import Path
from typing import Any, Dict, List
def _find_configs_dir(start: Path) -> Path:
    """向上查找第一个包含 configs/ 的目录"""
    for p in start.resolve().parents:
        candidate = p / "configs"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not find 'configs/' dir upwards from {start}"
    )
    
def manual_resolve(cfg: DictConfig, main_config_path: Path) -> DictConfig:
    """
    递归展开 *.yaml 字符串节点；
    写在 config 里的路径都当作相对于「main_config_path 所在目录下的 configs/」
    """
    # 统一锚点：main.yaml 同级目录下的 configs/
    anchor_dir = _find_configs_dir(main_config_path)
    if not anchor_dir.exists():
        raise FileNotFoundError(f"resolve anchor dir not found: {anchor_dir}")

    for key, value in list(cfg.items()):
        if isinstance(value, str) and value.endswith(('.yaml', '.yml')):
            sub_file = anchor_dir / value          # 自动拼到 configs/ 下面
            if not sub_file.exists():
                raise FileNotFoundError(f"sub config not found: {sub_file}")
            sub_cfg = OmegaConf.load(sub_file)
            sub_cfg = manual_resolve(sub_cfg, main_config_path)  # 递归子文件
            cfg[key] = sub_cfg
            print(f"Resolved config key '{key}' from file: {sub_file}")
        elif isinstance(value, DictConfig):
            cfg[key] = manual_resolve(value, main_config_path)  # 递归子节点
        else:
            pass  # 其他类型不处理
    return cfg



def get_abs_base_config_path(abs_config_path, base_config_path):
    # 使用pathlib处理路径
    abs_path = Path(abs_config_path)
    base_path = Path(base_config_path)
    
    # 确保abs_config_path包含configs
    abs_parts = abs_path.parts
    if 'configs' not in abs_parts:
        raise ValueError("abs_config_path must contain 'configs' directory")
    
    # 找到configs目录的位置
    configs_index = abs_parts.index('configs')
    
    # 构建configs目录的绝对路径
    configs_abs_dir = Path(*abs_parts[:configs_index + 1])
    
    # 处理base_config_path，移除开头的configs/
    base_parts = base_path.parts
    if base_parts and base_parts[0] == 'configs':
        base_relative = Path(*base_parts[1:])
    else:
        base_relative = base_path
    
    # 组合路径
    abs_base_config_path = configs_abs_dir / base_relative
    
    return str(abs_base_config_path.absolute())

def load_config(config_path):
    if config_path is not None and os.path.exists(config_path):
        config_path = Path(config_path)
        config = OmegaConf.load(config_path)
        config = manual_resolve(config, config_path)     # 传入主文件路径做锚点
    else:
        raise ValueError(f"Please provide a valid config path. check {config_path}")
    
    return config


def parse_unknown_to_dict(unknown: list[str]) -> dict[str, Any]:
    """
    像 argparse 一样把 unknown 解析成字典：
      --key val        -> {"key": "val"}
      --k v --flag     -> {"k": "v", "flag": "true"}
      --a.b.c 4        -> {"a.b.c": "4"}
    """
    result = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith('--'):
            i += 1
            continue
        
        key = token[2:]  # 剥掉 '--'
        if '=' in key:  # --key=val
            key, val = key.split('=', 1)
        elif i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
            val = unknown[i + 1]
            i += 1  # 跳过 value
        else:  # --flag
            val = "true"
        
        result[key] = val
        i += 1
    
    return result

def merge_dict_into_config(config: dict, flat: dict[str, Any]) -> None:
    """把扁平字典按点分路径展开并合并到 config。"""
    for key_path, val in flat.items():
        # 类型转换（能转才转，不硬转）
        if isinstance(val, str):
            val = _smart_cast(val)
        
        keys = key_path.split('.')
        cur = config
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = val
        

def _smart_cast(s: str) -> Any:
    """比 isdigit() 更聪明的转换，支持负数、科学计数法、bool。"""
    s = s.strip()
    if s == "true": return True
    if s == "false": return False
    
    # 先尝试 int（包括负数）
    try:
        return int(s)
    except ValueError:
        pass
    
    # 再尝试 float（支持 3e-5, -1.2 等）
    try:
        return float(s)
    except ValueError:
        pass
    
    return s

def patch_config_from_unknown(config, unknown: list[str]) -> None:
    """把 --a.b.c=123 这种 unknown 参数直接写进 config，能转数字就转。"""
    for token in unknown:
        if not token.startswith("--") or "=" not in token:
            continue
        
        key_path, val = token[2:].split("=", 1)
        keys = key_path.split(".")
        
        # 类型转换：数字就转，true/false 就转 bool，其余保持字符串
        if val.isdigit():
            val = int(val)
        elif val.replace(".", "", 1).isdigit():
            val = float(val)
        elif val == "true":
            val = True
        elif val == "false":
            val = False
        
        # 一路找下去，中间没有就建 dict
        cur = config
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        
        # 最后一级直接赋值
        cur[keys[-1]] = val
        