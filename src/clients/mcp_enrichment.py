"""
MCP enrichment service for lab history.

Encapsulates the full MCP enrichment flow: building observations from
recognition results, asking an LLM whether historical data is needed,
validating/normalizing MCP requests via terminology service, calling the
MCP server, and converting FHIR responses to RecognizedTable format.
"""

import asyncio
import json
import logging
import uuid
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateutil_parser
from jinja2 import Template

import src.utils as utils
from src.clients.mcp import McpClient
from src.clients.llm_client import LLMClient
from src.clients.terminology import TerminologyClient
from src.schemas.mcp import (
    AdditionalLabsPlan,
    Coding,
    DateFilter,
    InterpretationFilter,
    McpLabHistoryRequest,
    McpToolCallResult,
    NameFilter,
    SpecimenFilter,
)
from src.schemas.recognition import (
    RecognizedRow,
    RecognizedTable,
    ReportRecognitionResult,
)

logger = logging.getLogger()

LOINC_SYSTEM = "http://loinc.org"
NSI_SYSTEM = "https://nsi.rosminzdrav.ru"
SNOMED_SYSTEM = "http://snomed.info/sct"
HL7_SYSTEM = "http://hl7.org/fhir/observation-interpretation"


class McpEnrichmentService:
    """Self-contained service for MCP lab-history enrichment."""

    def __init__(
        self,
        mcp_client: McpClient,
        llm_client: LLMClient,
        terminology_client: TerminologyClient,
    ):
        self.llm_client = llm_client
        self.terminology_client = terminology_client
        self.mcp_client = mcp_client

    async def __call__(
        self,
        recognition_result: ReportRecognitionResult,
        topic_id: str,
        patient_info: Optional[Dict],
    ) -> Optional[List[RecognizedTable]]:
        """Full MCP enrichment flow.

        1. Build a compact observations summary from recognition_result.
        2. Ask an LLM whether historical lab data is needed (produces a plan).
        3. Normalize, validate, fetch via MCP, and convert results.

        Returns list of historical RecognizedTable items, or None on failure/skip.
        """
        try:
            observations = self._build_observations(recognition_result)
            if not observations:
                return None

            mcp_prompts = utils.PIPELINE_NAME2PROMPT.get("mcp_enrichment", {})
            mcp_config = utils.LLM_PIPELINE_CONFIGS.get("mcp_enrichment", {})

            base_prompt = (
                Template(mcp_prompts.get("base_prompt", "")).render(date=date.today()).strip()
            )

            user_prompt = (
                Template(mcp_prompts.get("template", ""))
                .render(
                    observations=observations,
                    patient_info=patient_info,
                )
                .strip()
            )

            plan = await self.llm_client.request(
                model_name=mcp_config.get("model_name"),
                response_format=AdditionalLabsPlan,
                base_prompt=base_prompt,
                user_prompt=user_prompt,
                temperature=mcp_config.get("temperature"),
                prompt_type="mcp_enrichment",
            )

            if plan is None:
                logger.warning("MCP LLM returned no plan")
                return None

            mcp_tables, _ = await self._process_enrichment(plan, topic_id)
            return mcp_tables if mcp_tables else None

        except Exception as e:
            logger.error(f"MCP enrichment failed: {e}", exc_info=True)
            return None

    @staticmethod
    def _extract_code_map_from_fhir(
        structured_output: Optional[Dict],
    ) -> Dict[Tuple[str, str], Dict[str, str]]:
        """Build a (table_name, labname, issued_date) → {code, codesystem} map.

        Delegates index building to utils.build_observation_index, which
        uses a two-pass algorithm over DiagnosticReport + Observation entries to
        produce a collision-free composite key. LOINC coding is preferred; falls
        back to the first available coding entry.
        """
        structured_output = structured_output or {}
        resources = (
            structured_output.get("resources") or structured_output.get("input_resources") or []
        )
        if not resources:
            return {}

        obs_index = utils.build_observation_index(resources)
        result: Dict[Tuple[str, str], Dict[str, str]] = {}

        for (table_name, labname, issued_date), obs_info in obs_index.items():
            coding_list = obs_info.get("coding") or []
            if not coding_list:
                continue
            best = next(
                (c for c in coding_list if c.get("system") == LOINC_SYSTEM),
                coding_list[0],
            )
            code = best.get("code", "")
            system = best.get("system", "")
            if code and system:
                result[(table_name, labname)] = {
                    "code": code,
                    "codesystem": system,
                }
        return result

    @staticmethod
    def _build_observations(
        recognition_result: ReportRecognitionResult,
    ) -> List[Dict]:
        """Build compact observations summary for the MCP enrichment prompt.

        Enriches each row with code/codesystem when available from the already-normalized
        FHIR resources in structured_output, so the LLM can echo them back in NameFilter
        and skip the terminology service lookup for known parameters.
        """
        code_map = McpEnrichmentService._extract_code_map_from_fhir(
            recognition_result.structured_output
        )
        observations_data = []
        for table in recognition_result.items:
            table_data = {
                "request_date": table.request_date or None,
                "result_date": table.result_date or None,
                "name": table.name or None,
                "biomaterial": table.biomaterial or None,
                "rows": [],
            }
            for row in table.rows or []:
                row_data = {
                    "labname": row.labname or "",
                    "result": row.result or "",
                    "measure": row.measure or None,
                    "ref_value": row.ref_value or None,
                }
                coding = code_map.get((table.name or "", row.labname or ""))
                if coding:
                    row_data["code"] = coding["code"]
                    row_data["codesystem"] = coding["codesystem"]
                table_data["rows"].append(row_data)
            observations_data.append(table_data)
        return observations_data

    @staticmethod
    def _unwrap_value_structure(obj: Any) -> Any:
        """Unwrap nested structures like {"value": ...} and return direct value."""
        if isinstance(obj, dict):
            if "value" in obj and len(obj) == 1:
                return McpEnrichmentService._unwrap_value_structure(obj["value"])
            return {
                key: McpEnrichmentService._unwrap_value_structure(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [McpEnrichmentService._unwrap_value_structure(item) for item in obj]
        return obj

    @staticmethod
    def parse_mcp_results_to_observations(
        mcp_results: List[McpToolCallResult],
    ) -> List[Dict[str, Any]]:
        """Parse MCP results and extract Observation resources.

        Returns list of Observation resources in entry format
        (with fullUrl and resource).
        """
        logger.info(f"Parsing MCP results: received {len(mcp_results)} results")

        if not mcp_results:
            logger.warning("MCP results list is empty")
            return []

        unwrap = McpEnrichmentService._unwrap_value_structure
        observations: List[Dict[str, Any]] = []

        for idx, mcp_result in enumerate(mcp_results):
            result_label = f"Result #{idx + 1}"

            # Step 1: extract structured_content or parse from text_content
            structured = mcp_result.structured_content

            if not isinstance(structured, dict):
                for text_idx, text_block in enumerate(mcp_result.text_content):
                    if not (
                        text_block.strip().startswith("{") or text_block.strip().startswith("[")
                    ):
                        continue
                    try:
                        candidate = unwrap(json.loads(text_block.strip()))
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(
                            f"{result_label}: failed to parse text_block[{text_idx}]: {e}"
                        )
                        continue

                    if isinstance(candidate, dict):
                        if "entry" in candidate and "resourceType" not in candidate:
                            candidate["resourceType"] = "Bundle"
                        structured = candidate
                        break

            if not isinstance(structured, dict):
                logger.warning(f"{result_label}: could not resolve structured dict, skipping")
                continue

            # Step 2: extract Observations depending on the resource type
            resource_type = structured.get("resourceType")

            if resource_type == "Bundle":
                entries = structured.get("entry", [])
                if not isinstance(entries, list) or not entries:
                    logger.warning(f"{result_label}: Bundle has no entries, skipping")
                    continue

                for entry in entries:
                    if not isinstance(entry, dict):
                        continue

                    resource = unwrap(entry.get("resource") or {})
                    if not isinstance(resource, dict):
                        continue

                    res_type = resource.get("resourceType")
                    if not res_type and all(
                        k in resource for k in ["code", "status", "category", "value"]
                    ):
                        res_type = "Observation"
                        resource["resourceType"] = "Observation"

                    if res_type == "Observation":
                        resource_id = resource.get("id") or str(uuid.uuid4())
                        observations.append(
                            {"fullUrl": f"urn:uuid:{resource_id}", "resource": resource}
                        )

            elif resource_type == "Observation":
                obs_id = structured.get("id") or str(uuid.uuid4())
                observations.append({"fullUrl": f"urn:uuid:{obs_id}", "resource": structured})

        return observations

    def _convert_observations_to_tables(
        self,
        mcp_results: List[McpToolCallResult],
        code_to_display: Optional[Dict[str, str]] = None,
    ) -> List[RecognizedTable]:
        """Convert MCP tool call results to RecognizedTable format.

        Parses FHIR Observation resources from MCP results, groups them by
        effective date, and returns a list of RecognizedTable objects.
        """
        observations = self.parse_mcp_results_to_observations(mcp_results)
        if not observations:
            return []

        tables_dict: Dict[str, Dict] = {}

        for obs_entry in observations:
            obs: Dict[str, Any] = obs_entry["resource"]

            # Date
            raw_date = obs.get("effectiveDateTime") or obs.get("issued", "")
            try:
                date_str = (
                    dateutil_parser.isoparse(raw_date).date().strftime("%Y-%m-%d")
                    if raw_date
                    else ""
                )
            except Exception:
                date_str = raw_date[:10] if len(raw_date) >= 10 else ""

            table_key = date_str or "unknown"
            if table_key not in tables_dict:
                tables_dict[table_key] = {
                    "request_date": date_str,
                    "result_date": date_str,
                    "rows": [],
                }

            # Labname
            codings: List[Dict] = obs.get("code", {}).get("coding", [])

            obs_code = next(
                (c.get("code") for c in codings if c.get("system") == LOINC_SYSTEM),
                None,
            ) or next(
                (c.get("code") for c in codings if c.get("code")),
                None,
            )

            if code_to_display and obs_code and obs_code in code_to_display:
                labname = code_to_display[obs_code]
            else:
                labname = (
                    next(
                        (
                            c["display"]
                            for c in codings
                            if c.get("system", "").startswith(NSI_SYSTEM) and c.get("display")
                        ),
                        None,
                    )
                    or next((c["display"] for c in codings if c.get("display")), None)
                    or obs.get("code", {}).get("text", "")
                )

            # Value and units
            value_quantity = obs.get("valueQuantity") or {}
            value = value_quantity.get("value") if value_quantity else obs.get("valueString", "")
            measure = value_quantity.get("unit", "")

            # Reference range
            ref_range = obs.get("referenceRange") or []
            ref_val = ""
            if ref_range:
                low = ref_range[0].get("low", {}).get("value")
                high = ref_range[0].get("high", {}).get("value")
                if low is not None and high is not None:
                    ref_val = f"{low} - {high}"
                elif high is not None:
                    ref_val = f"<= {high}"
                elif low is not None:
                    ref_val = f">= {low}"

            tables_dict[table_key]["rows"].append(
                RecognizedRow(
                    labname=labname or "",
                    result=str(value) if value is not None else "",
                    measure=measure,
                    ref_value=ref_val,
                    comment="",
                )
            )

        return [
            RecognizedTable(
                request_date=td["request_date"],
                result_date=td["result_date"],
                rows=td["rows"],
            )
            for td in tables_dict.values()
        ]

    @staticmethod
    def _validate_requests(
        requests: List[McpLabHistoryRequest],
    ) -> List[McpLabHistoryRequest]:
        """Validate MCP requests: filter out invalid filters and requests.

        - Skip DateFilter if both dates are None
        - Skip NameFilter if code is missing
        - Ensure Coding objects have system (default to LOINC if missing)
        - Filter out requests that contain only Date filters
        """
        validated_requests = []
        for idx, req in enumerate(requests):
            validated_filters = []
            has_non_date_filter = False

            for filter_item in req.filters:
                if isinstance(filter_item, DateFilter):
                    if filter_item.date_from is None and filter_item.date_to is None:
                        continue
                    validated_filters.append(filter_item)
                elif isinstance(filter_item, NameFilter):
                    if not filter_item.name.code or filter_item.name.code == "":
                        logger.warning(
                            f"Request {idx + 1}: Skipping NameFilter without code: "
                            f"{filter_item.name.display}"
                        )
                        continue
                    if not filter_item.name.system:
                        filter_item.name = Coding(
                            code=filter_item.name.code,
                            codeSystemUrl=LOINC_SYSTEM,
                            display=filter_item.name.display,
                        )
                    has_non_date_filter = True
                    validated_filters.append(filter_item)
                elif isinstance(filter_item, InterpretationFilter):
                    if not filter_item.interpretation.system:
                        filter_item.interpretation = Coding(
                            code=filter_item.interpretation.code,
                            codeSystemUrl=HL7_SYSTEM,
                            display=filter_item.interpretation.display,
                        )
                    has_non_date_filter = True
                    validated_filters.append(filter_item)
                elif isinstance(filter_item, SpecimenFilter):
                    if not filter_item.specimen.system:
                        filter_item.specimen = Coding(
                            code=filter_item.specimen.code,
                            codeSystemUrl=SNOMED_SYSTEM,
                            display=filter_item.specimen.display,
                        )
                    has_non_date_filter = True
                    validated_filters.append(filter_item)
                else:
                    validated_filters.append(filter_item)

            if not has_non_date_filter:
                logger.warning(
                    f"Request {idx + 1} contains only Date filters and will be "
                    f"skipped. Request: {req.model_dump(by_alias=True, exclude_none=True)}"
                )
                continue

            validated_requests.append(
                McpLabHistoryRequest(topicId=req.topicId, filters=validated_filters)
            )

        if len(validated_requests) < len(requests):
            logger.info(
                f"Filtered out {len(requests) - len(validated_requests)} invalid "
                f"requests. Remaining: {len(validated_requests)} requests"
            )

        return validated_requests

    async def _normalize_display(self, display: str):
        try:
            results = await self.terminology_client.search_loinc(display)
            return display, results
        except Exception as e:
            logger.exception(
                f"Error normalizing '{display}': {e}",
            )
            return display, None

    async def _normalize_requests_with_terminology(
        self,
        requests: List[McpLabHistoryRequest],
    ) -> Tuple[List[McpLabHistoryRequest], Dict[str, str]]:
        """Normalize MCP requests: find LOINC codes for NameFilter via terminology.

        For each NameFilter without a code, searches the terminology service
        by display name and fills code + codeSystemUrl.

        Returns (normalized requests, code_to_display mapping).
        """
        code_to_display: Dict[str, str] = {}
        names_to_normalize: list[tuple[int, int, str]] = []

        for req_idx, req in enumerate(requests):
            for filter_idx, filter_item in enumerate(req.filters):
                if not isinstance(filter_item, NameFilter):
                    continue

                name = filter_item.name
                code = name.code or ""
                display = name.display or ""
                system = name.system or ""

                if code:
                    if system and display:
                        code_to_display[code] = display
                    continue

                if display:
                    names_to_normalize.append((req_idx, filter_idx, display))

        if names_to_normalize:
            logger.debug(
                f"Normalizing {len(names_to_normalize)} lab names through terminology service",
            )

            unique_displays = {display for _, _, display in names_to_normalize}

            normalization_results = await asyncio.gather(
                *(self._normalize_display(display) for display in unique_displays)
            )

            normalized_map = {display: results for display, results in normalization_results}

            for req_idx, filter_idx, display in names_to_normalize:
                results = normalized_map.get(display)

                if not results:
                    logger.warning(f"No LOINC code found for '{display}'")
                    continue

                first_result = results[0]

                loinc_code = first_result.get("code")
                if not loinc_code:
                    continue

                loinc_display = first_result.get("display") or display

                req = requests[req_idx]
                name_filter = req.filters[filter_idx]

                if not isinstance(name_filter, NameFilter):
                    continue

                name_filter.name = Coding(
                    code=loinc_code,
                    codeSystemUrl=LOINC_SYSTEM,
                    display=loinc_display,
                )

                code_to_display[loinc_code] = display

                logger.debug(
                    f"Normalized '{display}' -> LOINC {loinc_code} ({loinc_display})",
                )

        return requests, code_to_display

    async def _process_enrichment(
        self,
        plan: AdditionalLabsPlan,
        topic_id: str,
    ) -> Tuple[List[RecognizedTable], Dict[str, str]]:
        """Orchestrate MCP enrichment after the LLM plan is ready.

        1. Set topic_id for each request
        2. Normalize requests with terminology service
        3. Validate requests
        4. Call MCP tool
        5. Convert MCP results to RecognizedTable format

        Returns (list of historical RecognizedTable, code_to_display mapping).
        """
        if not plan.need_tool or not plan.requests:
            logger.debug("MCP tool not needed according to LLM")
            return [], {}

        for req in plan.requests:
            req.topicId = topic_id

        requests_with_loinc, code_to_display = await self._normalize_requests_with_terminology(
            plan.requests
        )

        validated_requests = self._validate_requests(requests_with_loinc)
        logger.debug(f"validated_requests: {validated_requests}")

        if not validated_requests:
            logger.warning("All MCP requests were filtered out. Skipping MCP enrichment.")
            return [], code_to_display

        try:
            tool_name, mcp_results = await self.mcp_client.call_lab_history(
                requests=validated_requests,
            )

            mcp_tables = self._convert_observations_to_tables(mcp_results, code_to_display)
            if mcp_tables:
                logger.info(
                    f"MCP enrichment completed: {len(mcp_tables)} tables, "
                    f"{sum(len(t.rows or []) for t in mcp_tables)} observations"
                )
            else:
                logger.warning("MCP returned no observations to add")
            return mcp_tables, code_to_display

        except Exception as e:
            logger.error(f"Error during MCP enrichment: {e}", exc_info=True)
            return [], code_to_display
