import os
from typing import Dict

import yaml


def load_config(config_path: str) -> Dict:
    """
    Loads a YAML configuration file.
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)
