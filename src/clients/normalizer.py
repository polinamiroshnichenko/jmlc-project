import hashlib
import json
import logging
from typing import Dict, List, Tuple

from jinja2 import Template

import src.utils as utils
from src.clients.llm_client import LLMClient
from src.clients.terminology import TerminologyClient
from src.ports.assets_repository import AssetsRepository
from src.schemas.recognition import (
    LoincSearchVariantsResult,
    RecognizedTable,
    ReportNormalizedResult,
)

logger = logging.getLogger("ray.serve")


def generate_hash_code(table_name: str, labname: str) -> str:
    combined = f"{table_name}|{labname}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


class LabTestRecognitionNormalizer:
    """Normalizer for laboratory test recognition results (LOINC/UCUM enrichment)."""

    def __init__(
        self,
        llm_client: LLMClient,
        terminology_client: TerminologyClient,
        assets: AssetsRepository,
    ):
        self.llm_client = llm_client
        self.terminology_client = terminology_client
        self._assets = assets

    def _form_loinc_variants_prompt(self, labnames: List[Dict[str, str]]) -> Tuple[str, str]:
        prompts = utils.PIPELINE_NAME2PROMPT.get("loinc_search_variants", {})
        base_prompt = prompts.get("base_prompt", "").strip()
        user_prompt = Template(prompts.get("template", "")).render({"labnames": labnames}).strip()
        return base_prompt, user_prompt

    async def _generate_loinc_search_variants_batch(
        self, labnames: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        if not labnames:
            return {}
        try:
            base_prompt, user_prompt = self._form_loinc_variants_prompt(labnames)
            config = utils.LLM_PIPELINE_CONFIGS["loinc_search_variants"]
            result = await self.llm_client.request(
                model_name=config["model_name"],
                response_format=LoincSearchVariantsResult,
                base_prompt=base_prompt,
                user_prompt=user_prompt,
                temperature=config["temperature"],
                prompt_type="loinc_search_variants",
            )
            if result is None or not result.items:
                logger.warning("Failed to generate LOINC search variants batch")
                return {}
            return result.items
        except Exception as e:
            logger.error(f"Error generating LOINC search variants batch: {e}")
            return {}

    async def _collect_all_variants_and_search_terminology(
        self, tables: List[RecognizedTable]
    ) -> Tuple[Dict[str, List[Dict]], Dict[Tuple[str, str], List[str]]]:
        labnames_list = [
            {"table_name": table.name or "", "labname": row.labname}
            for table in tables
            for row in (table.rows or [])
            if row.labname
        ]

        if not labnames_list:
            return {}, {}

        logger.info(f"Generating LOINC variants for {len(labnames_list)} labnames")
        variants_dict = await self._generate_loinc_search_variants_batch(labnames_list)

        labname_to_variants_map: Dict[Tuple[str, str], List[str]] = {}
        unique_variants: List[str] = []
        seen: set = set()

        for key, variants in variants_dict.items():
            parts = key.split("|", 1)
            if len(parts) != 2:
                continue
            labname_to_variants_map[(parts[0], parts[1])] = variants
            for v in variants:
                if v not in seen:
                    unique_variants.append(v)
                    seen.add(v)

        if not unique_variants:
            return {}, labname_to_variants_map

        logger.info(f"Searching terminology for {len(unique_variants)} variants")
        results = await self.terminology_client.search_loinc_multiple(unique_variants)
        filtered = {k: v for k, v in results.items() if v}
        logger.info(f"Got terminology results for {len(filtered)}/{len(unique_variants)} variants")
        return filtered, labname_to_variants_map

    def _form_prompt(
        self,
        tables: List[RecognizedTable],
        terminology_results: Dict[str, List[Dict]],
        labname_to_variants_map: Dict[Tuple[str, str], List[str]],
    ) -> Tuple[str, str]:
        tables_json = [table.dict() for table in tables]
        labnames: Dict[str, List[Dict]] = {}

        for table in tables:
            table_name = table.name or ""
            for row in table.rows or []:
                if not row.labname:
                    continue
                variants = labname_to_variants_map.get((table_name, row.labname), [])
                available_codes = []
                for variant in variants:
                    for res in terminology_results.get(variant) or []:
                        if not isinstance(res, dict):
                            continue
                        code = res.get("code", "")
                        display_names = [
                            d.get("content", "")
                            for d in res.get("designations", [])
                            if isinstance(d, dict) and d.get("content")
                        ]
                        if code and display_names:
                            available_codes.append(
                                {
                                    "code": code,
                                    "display": display_names[0],
                                    "all_display_names": display_names,
                                }
                            )
                if available_codes:
                    labnames[f"{table_name}|{row.labname}"] = available_codes

        ucum_records = self._assets.ucum_codes()
        data = {
            "tables": tables_json,
            "biomaterials": self._assets.biomaterials().keys(),
            "ucum_codes": ucum_records.model_dump(by_alias=True),
            "labnames": json.dumps(labnames, ensure_ascii=False, indent=2),
        }
        prompts = utils.PIPELINE_NAME2PROMPT.get("recognition_normalization", {})
        base_prompt = Template(prompts.get("base_prompt", "")).render(data).strip()
        user_prompt = Template(prompts.get("template", "")).render(data).strip()
        return base_prompt, user_prompt

    def _postprocess(
        self, normalized: ReportNormalizedResult, original_tables: List[RecognizedTable]
    ) -> ReportNormalizedResult:
        external_codesystem = "http://ml.example.com/laboratory/external-test-code"
        for table_idx, table in enumerate(normalized.items):
            if table_idx >= len(original_tables) or not table.rows:
                continue
            original_table = original_tables[table_idx]
            table_name = original_table.name or ""
            original_rows = original_table.rows or []
            for row_idx, row in enumerate(table.rows):
                if row_idx >= len(original_rows):
                    continue
                if getattr(row, "code", ""):
                    continue
                original_labname = original_rows[row_idx].labname or ""
                row.code = generate_hash_code(table_name, original_labname)
                row.codesystem = external_codesystem
                if not row.display:
                    row.display = original_labname
                if not row.text:
                    row.text = original_labname
        return normalized

    async def __call__(self, tables: List[RecognizedTable]) -> ReportNormalizedResult:
        if not tables:
            return ReportNormalizedResult(items=[])

        terminology_results, labname_to_variants_map = (
            await self._collect_all_variants_and_search_terminology(tables)
        )

        base_prompt, user_prompt = self._form_prompt(
            tables, terminology_results, labname_to_variants_map
        )

        config = utils.LLM_PIPELINE_CONFIGS["recognition_normalization"]
        result = await self.llm_client.request(
            model_name=config["model_name"],
            response_format=ReportNormalizedResult,
            base_prompt=base_prompt,
            user_prompt=user_prompt,
            temperature=config["temperature"],
            prompt_type="recognition_normalization",
        )

        if result is None:
            logger.error("LLM returned no result for lab test normalization")
            return ReportNormalizedResult(items=[])

        return self._postprocess(result, tables)
