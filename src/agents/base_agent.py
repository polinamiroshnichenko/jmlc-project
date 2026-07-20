from datetime import datetime
from typing import Any, Dict

import pytz
from jinja2 import Template

from src.agents.config_loader import (
    load_agent_base_prompt,
    load_agent_config,
    load_agent_template,
)
from src.schemas.common import Patient
import src.utils as utils


class BaseAgent:
    config_name: str  # set by agent subclass

    def __init__(self):
        agent_cfg = load_agent_config(self.config_name)
        self.llm_request_params: Dict[str, Any] = agent_cfg["request_params"]
        self._base_template = Template(load_agent_base_prompt(self.config_name))
        self._body_template = Template(load_agent_template(self.config_name))

    @staticmethod
    def _build_patient_context(patient: Patient) -> Dict[str, Any]:
        moscow_tz = pytz.timezone("Europe/Moscow")
        age = patient.age
        if not age and patient.date_of_birth:
            age = str(utils.calculate_age(patient.date_of_birth))

        return {
            "current_date": datetime.now(moscow_tz).strftime("%Y-%m-%d"),
            "gender": {"male": "Мужской", "female": "Женский"}.get(patient.gender or "", ""),
            "age": age or "",
        }
