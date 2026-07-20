import re
from datetime import date

from src.schemas import chat as chat_schemas
from src.schemas import common as common_schemas


from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel
from src.config import config

SYSTEM_TO_URL = {
    "LOINC": "https://loinc.org/",
    "SCTID": "http://snomed.info/sct",
    "UCUM": "http://unitsofmeasure.org",
    "FSLI": "https://nsi.rosminzdrav.ru/",
    "STEX": "https://terminology.example.com/snomed-extensions",
    "804N": "https://org.gnicpm.ru/wp-content/uploads/2019/01/Prikaz-Minzdrava-Rossii-ot-13.10.2017-N-804n-s-izm.-ot-12.0.pdf",
}

PIPELINE_NAME2PROMPT = {}
prompts_folder = config.root_dir / "prompts"

for prompt_file in prompts_folder.glob("*.yml"):
    with prompt_file.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        if not isinstance(data, dict):
            continue
        pipeline_name = prompt_file.stem
        prompts_dict = {}
        for key, value in data.items():
            if isinstance(value, str):
                prompts_dict[key] = value.strip()
            else:
                prompts_dict[key] = value
        PIPELINE_NAME2PROMPT[pipeline_name] = prompts_dict


ALLOWED_MIME_TYPES = [
    "text/plain",
    "application/json",
    "application/pdf",
    "image/heic",
    "image/png",
    "image/jpeg",
]

LLM_PIPELINE_CONFIGS = {
    "prescription_normalization": {
        "model_name": "",
        "temperature": 0.1,
    },
    "recognition_normalization": {
        "model_name": "",
        "temperature": 0.1,
    },
    "prescription_recognition": {
        "model_name": "",
        "temperature": 0.1,
    },
    "report_recognition": {
        "model_name": "",
        "temperature": 1.0,
    },
    "report_interpretation": {
        "model_name": "",
        "temperature": 1.0,
        "timeout": 600,
        "max_tokens": 8000,
        "retry_count": 1,
    },
    "loinc_search_variants": {
        "model_name": "",
        "temperature": 0.1,
    },
    "translation": {
        "model_name": "",
        "temperature": 0.3,
    },
    "mcp_enrichment": {
        "model_name": "",
        "temperature": 0.2,
    },
}


def to_plain(obj):
    if isinstance(obj, BaseModel):
        return {k: to_plain(v) for k, v in obj.model_dump(exclude_none=True, by_alias=True).items()}
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        dt = obj.astimezone(timezone.utc)
        formatted = dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        return f"{formatted[:-2]}:{formatted[-2:]}"
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _parse_fhir_reference(ref: str) -> str:
    """Extract resource ID from a FHIR reference string.

    Handles both 'urn:uuid:<id>' and 'ResourceType/<id>' formats.
    """
    if ":" in ref:
        return ref.split(":")[-1]
    return ref.split("/")[-1]


def format_fhir_date(value: str) -> str:
    """Normalize FHIR date/dateTime strings to YYYY-MM-DD."""
    return value[:10] if value else ""


def build_observation_index(
    resources: Optional[List[Dict[str, Any]]],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Build (table_name, labname, issued_date) -> {id, coding} index from FHIR Bundle resources."""
    if not resources:
        return {}

    index = {}
    for bundle in resources:
        entries = bundle.get("entry", []) if isinstance(bundle, dict) else []

        # Pass 1: map Observation IDs to all linked DiagnosticReport names/dates and resource
        obs_id2reports: dict[str, list[tuple[str, str]]] = defaultdict(list)
        obs_id2resource: dict[str, dict] = {}

        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type == "DiagnosticReport":
                code = resource.get("code", {})
                report_name = code.get("text", "") or code.get("coding", [{}])[0].get("display", "")
                report_issued_date = format_fhir_date(
                    resource.get("issued", "") or resource.get("effectiveDateTime", "")
                )
                for ref in resource.get("result", []):
                    obs_id = _parse_fhir_reference(ref.get("reference", ""))
                    obs_id2reports[obs_id].append((report_name, report_issued_date))

            elif resource_type == "Observation":
                obs_id = resource.get("id", "")
                if obs_id:
                    obs_id2resource[obs_id] = resource

        # Pass 2: build (table_name, lab_name, issued_date) -> {id, coding} index
        for obs_id, observation in obs_id2resource.items():
            labname = observation.get("code", {}).get("text", "")
            if labname:
                observation_issued_date = format_fhir_date(
                    observation.get("issued", "") or observation.get("effectiveDateTime", "")
                )
                reports = obs_id2reports.get(obs_id) or [("", "")]
                for table_name, report_issued_date in reports:
                    issued_date = observation_issued_date or report_issued_date
                    index[(table_name, labname, issued_date)] = {
                        "id": obs_id,
                        "coding": observation.get("code", {}).get("coding", []),
                    }

    return index


def calculate_age(birth_date: date) -> int:
    today = date.today()
    return (
        today.year
        - birth_date.year
        - ((today.month, today.day) < (birth_date.month, birth_date.day))
    )


def convert_patient(patient: chat_schemas.Patient) -> common_schemas.Patient:
    gender = patient.gender
    return common_schemas.Patient(
        id=patient.id,
        gender=gender.lower() if gender else None,
        date_of_birth=patient.birth_date,
        age=patient.age,
        is_pregnant=patient.is_pregnant,
    )


def remove_agent_metadata(text: str) -> str:
    """Removes service tags like "[Агент: AgentType.general_assistant]" from the text."""
    if not text:
        return text

    cleaned = re.sub(r"\[Агент: [^\]]+\]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
