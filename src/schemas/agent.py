from typing import Dict, List, Optional

from jinja2 import Template
from pydantic import BaseModel


class CDSAgentConfig(BaseModel):
    name: str
    llm: str
    response_format: Optional[str]
    pipeline: str
    hook: str
    description: str
    prefetch: Dict[str, str]
    base_prompt: str
    template: str
    examples: List[str] = []
    base_prompt_template: Optional[Template] = None
    body_template: Optional[Template] = None
