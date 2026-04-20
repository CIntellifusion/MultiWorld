from importlib import import_module
from omegaconf import OmegaConf,DictConfig

def import_class_from_string(class_path: str):
    module_path, class_name = class_path.rsplit('.', 1)
    module = import_module(module_path)
    cls = getattr(module, class_name)
    return cls

def instantiate_from_config(config: OmegaConf) -> object:
    if 'target' not in config:
        raise ValueError("Config must contain 'target' key specifying the class path.")
    
    class_path = config.pop('target')
    cls = import_class_from_string(class_path)
    instance = cls(**config.params)
    return instance