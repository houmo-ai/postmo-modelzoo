#!/usr/bin/env python3

import yaml


def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


def dump_yaml(data: str):
    return yaml.dump(data, allow_unicode=True, default_flow_style=False)
