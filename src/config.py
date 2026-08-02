"""Loads config.yaml + .env and exposes one CFG object everywhere."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for a config path entry; creates the folder."""
    p = ROOT / CFG["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)
