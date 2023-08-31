import sys
import re
import fnmatch
from collections import defaultdict, deque
from copy import deepcopy
from ._pretrained import PretrainedCfg, DefaultCfg, split_model_name_tag
from dataclasses import replace
from typing import Optional

_model_entrypoints = {}  # mapping of model names to entrypoint fns
_onnx_entrypoints = {} 

_model_default_cfgs = {} 
_model_pretrained_cfgs = {} 
_model_has_pretrained = set() 
_model_with_tags = defaultdict(list)

def register_model(model_class):
    mod = sys.modules[model_class.__module__]

    model_name = model_class.__name__
    _model_entrypoints[model_name] = model_class

    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        default_cfg = mod.default_cfgs[model_name]

        if not isinstance(default_cfg, DefaultCfg):
            # new style default cfg dataclass w/ multiple entries per model-arch
            assert isinstance(default_cfg, dict)
            # old style cfg dict per model-arch
            pretrained_cfg = PretrainedCfg(**default_cfg)
            default_cfg = DefaultCfg(tags=deque(['']), cfgs={'': pretrained_cfg})

        for tag_idx, tag in enumerate(default_cfg.tags):
            is_default = tag_idx == 0
            pretrained_cfg = default_cfg.cfgs[tag]
            model_name_tag = '.'.join([model_name, tag]) if tag else model_name
            replace_items = dict(architecture=model_name, tag=tag if tag else None)
            if pretrained_cfg.hf_hub_id and pretrained_cfg.hf_hub_id == 'timm/':
                # auto-complete hub name w/ architecture.tag
                replace_items['hf_hub_id'] = pretrained_cfg.hf_hub_id + model_name_tag
            pretrained_cfg = replace(pretrained_cfg, **replace_items)

            if is_default:
                _model_pretrained_cfgs[model_name] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    # add tagless entry if it's default and has weights
                    _model_has_pretrained.add(model_name)

            if tag:
                _model_pretrained_cfgs[model_name_tag] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    # add model w/ tag if tag is valid
                    _model_has_pretrained.add(model_name_tag)
                _model_with_tags[model_name].append(model_name_tag)
            else:
                _model_with_tags[model_name].append(model_name)  # has empty tag (to slowly remove these instances)

        _model_default_cfgs[model_name] = default_cfg

    return model_class

def register_onnx(onnx):
    model_name = onnx.__name__
    _onnx_entrypoints[model_name] = onnx
    return onnx

def is_model(model_name):
    """ Check if a model name exists
    """
    return model_name in _model_entrypoints

def is_onnx(model_name):
    """ Check if a onnx model name exists
    """
    return model_name in _onnx_entrypoints

def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]

def onnx_entrypoint(model_name):
    """Fetch a onnx entrypoint for specified model name
    """
    return _onnx_entrypoints[model_name]

def get_pretrained_cfg(model_name: str, allow_unregistered: bool = True) -> Optional[PretrainedCfg]:
    if model_name in _model_pretrained_cfgs:
        return deepcopy(_model_pretrained_cfgs[model_name])
    arch_name, tag = split_model_name_tag(model_name)
    if arch_name in _model_default_cfgs:
        # if model arch exists, but the tag is wrong, error out
        raise RuntimeError(f'Invalid pretrained tag ({tag}) for {arch_name}.')
    if allow_unregistered:
        # if model arch doesn't exist, it has no pretrained_cfg registered, allow a default to be created
        return None
    raise RuntimeError(f'Model architecture ({arch_name}) has no pretrained cfg registered.')
