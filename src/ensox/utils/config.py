import os
import re

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_string(value):
    def repl(match):
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return match.group(0)

    return os.path.expanduser(_ENV_PATTERN.sub(repl, value))


def _expand_env_tree(obj):
    if isinstance(obj, dict):
        return {k: _expand_env_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_tree(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_string(obj)
    return obj


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _expand_env_tree(data)
