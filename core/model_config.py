# -*- coding:utf-8 -*-

"""
@date: 2022/8/23 下午4:14
@summary: 模型文件配置加载
"""
import os
import yaml
from core.system_config import project_dir

class AttrDict:
    """
    Attr dict: make value private
    """
    def __init__(self, d):
        self.dict = d

    def __getattr__(self, attr):
        value = self.dict[attr]
        if isinstance(value, dict):
            return AttrDict(value)
        else:
            return value
    def __str__(self):
        return str(self.dict)

def get_model_config(model_name):
    """获取模型配置"""
    config_file = os.path.join(project_dir, f"etc/{model_name}.yaml")
    with open(config_file, encoding='utf-8') as f:
        if hasattr(yaml, 'FullLoader'):
            config = yaml.load(f, Loader=yaml.FullLoader)
        elif hasattr(yaml, 'CLoader'):
            config = yaml.load(f, Loader=yaml.CLoader)
        elif hasattr(yaml, 'UnsafeLoader'):
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        else:
            raise Exception(" error file !")
    config = AttrDict(config)
    return config
