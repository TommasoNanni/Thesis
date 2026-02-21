import yaml
from pathlib import Path

def load_config(filename):
    path = Path(filename)

    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config("configuration/config.yaml")