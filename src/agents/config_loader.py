import os
from pathlib import Path
from typing import Dict

import yaml

_CONFIGS_DIR = Path(os.getenv("CONFIGS_DIR", str(Path(__file__).parent.parent / "configs")))


def _load_yaml(path: Path) -> Dict:
    with open(path, "rb") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(agent_name: str) -> Dict:
    return _load_yaml(_CONFIGS_DIR / agent_name / "config.yml")


def load_agent_base_prompt(agent_name: str) -> str:
    prompt_dir = _CONFIGS_DIR / agent_name
    files = sorted(prompt_dir.glob("[0-9][0-9]_*.md"))
    return "\n\n".join(p.read_text(encoding="utf-8").strip() for p in files)


def load_agent_template(agent_name: str) -> str:
    path = _CONFIGS_DIR / agent_name / "context.md"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""
