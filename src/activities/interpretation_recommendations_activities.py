from typing import Callable, Dict, List, Optional

from pydantic import BaseModel
from temporalio import activity

from src.clients.fhir_converter import FhirToReportResultConverter
from src.ports.assets_repository import AssetsRepository
from src.schemas.chat import InitiatorType
from src.schemas.recognition import ReportRecognitionResult
from src.schemas.llm import ChatMessage
from src.utils import PIPELINE_NAME2PROMPT


class BuildCheckupDialogProps(BaseModel):
    interpretation_result: Dict
    initiator: Optional[InitiatorType] = None
    recognition_result: Optional[ReportRecognitionResult] = None


class AnnotateRecomendationsReassignedProps(BaseModel):
    checkup_response: Dict
    recognition_result: ReportRecognitionResult


class InterpretationRecommendationsActivities:

    def __init__(self, fhir_converter: FhirToReportResultConverter, assets: AssetsRepository):
        self.fhir_converter = fhir_converter
        self._assets = assets

    def get_activities(self) -> list[Callable]:
        return [
            self.get_checkup_prompts,
            self.extract_los_client_metadata,
            self.build_checkup_dialog,
            self.build_checkup_interpretation_text,
            self.build_los_client_reassign_context,
            self.compute_reassigned_value,
            self.convert_dialog_to_chat_messages,
            self.format_los_client_metadata_section,
            self.annotate_recommendations_reassigned,
        ]

    @staticmethod
    def _normalize_text(value: str) -> str:
        return (value or "").strip().lower()

    @staticmethod
    def _unique_preserve_order(values: List[str]) -> List[str]:
        return list(dict.fromkeys(v for v in values if v))

    @activity.defn(name="get_checkup_prompts")
    async def get_checkup_prompts(self, initiator: Optional[InitiatorType] = None) -> Dict:
        """Get checkup prompts based on initiator type"""
        LOS_CLIENT_INITIATOR: InitiatorType = "Los_client"
        CHECKUP_ASSISTANT_PROMPT_DEFAULT = "checkup_assistant"
        CHECKUP_ASSISTANT_PROMPT_LOS_CLIENT = "checkup_assistant_los_client"

        prompt_key = (
            CHECKUP_ASSISTANT_PROMPT_LOS_CLIENT
            if initiator == LOS_CLIENT_INITIATOR
            else CHECKUP_ASSISTANT_PROMPT_DEFAULT
        )
        return PIPELINE_NAME2PROMPT.get(prompt_key, {})

    @activity.defn(name="extract_los_client_metadata")
    async def extract_los_client_metadata(
        self, recognition_result: ReportRecognitionResult
    ) -> Dict[str, List[str]]:
        """Extract biomaterials and tube types for LOS client"""
        biomaterials = [
            table.biomaterial for table in recognition_result.items or [] if table.biomaterial
        ]

        tube_types: List[str] = []
        for bundle in (recognition_result.structured_output or {}).get("input_resources") or []:
            if isinstance(bundle, dict):
                tube_types.extend(self.fhir_converter.extract_tube_types_from_bundle(bundle))

        return {
            "biomaterials": self._unique_preserve_order(biomaterials),
            "tube_types": self._unique_preserve_order(tube_types),
        }

    @activity.defn(name="build_los_client_reassign_context")
    async def build_los_client_reassign_context(
        self, recognition_result: ReportRecognitionResult
    ) -> Dict[str, set]:
        """
        Collect (biomaterial, tube) pairs and already-performed hxids from the
        input FHIR bundles, used to decide whether a recommendation can be
        re-drawn from the already-collected sample.
        """
        pairs: set = set()
        bundle_hxids: set = set()

        for bundle in (recognition_result.structured_output or {}).get("input_resources") or []:
            if not isinstance(bundle, dict):
                continue
            for (
                biomaterial,
                tube_type,
            ) in self.fhir_converter.extract_specimen_container_pairs_from_bundle(bundle):
                pairs.add((self._normalize_text(biomaterial), self._normalize_text(tube_type)))
            for hxid in self.fhir_converter.extract_report_hxids_from_bundle(bundle):
                if hxid:
                    bundle_hxids.add(hxid.strip())

        return {"pairs": pairs, "bundle_hxids": bundle_hxids}

    @activity.defn(name="compute_reassigned_value")
    async def compute_reassigned_value(self, rec_hxids: set, los_context: Dict[str, set]) -> str:
        """Compute whether a recommendation can be reassigned"""
        catalog = self._assets.specimen_container_catalog_mapping()
        if not catalog:
            return "unavailable"

        bundle_hxids = los_context.get("bundle_hxids", set())
        pairs = los_context.get("pairs", set())

        for biomaterial, containers in catalog.items():
            if not isinstance(containers, dict):
                continue
            biomaterial_norm = self._normalize_text(biomaterial)
            for tube_type, analyses in containers.items():
                if (biomaterial_norm, self._normalize_text(tube_type)) not in pairs:
                    continue
                for analysis in analyses or []:
                    hxid = (
                        analysis.hxid if hasattr(analysis, "hxid") else (analysis.get("hxid") or "")
                    ).strip()
                    if hxid and hxid in rec_hxids:
                        # Available only if it can be re-drawn and is not already done.
                        return "unavailable" if hxid in bundle_hxids else "available"

        return "unavailable"

    @activity.defn(name="annotate_recommendations_reassigned")
    async def annotate_recommendations_reassigned(
        self, props: AnnotateRecomendationsReassignedProps
    ) -> None:
        """
        Annotate each recommendation with `reassigned` while the input bundles
        (recognition_result.structured_output) are still available.
        """
        if not isinstance(props.checkup_response, dict):
            return
        recommendations = props.checkup_response.get("recommendations") or []
        if not recommendations:
            return

        los_context = await self.build_los_client_reassign_context(props.recognition_result)
        for rec_data in recommendations:
            if not isinstance(rec_data, dict):
                continue
            rec_hxids = {
                (hxid.get("code") or "").strip()
                for hxid in rec_data.get("hxids") or []
                if (hxid.get("code") or "").strip()
            }
            rec_data["reassigned"] = await self.compute_reassigned_value(rec_hxids, los_context)

    @activity.defn(name="format_los_client_metadata_section")
    async def format_los_client_metadata_section(self, metadata: Dict[str, List[str]]) -> str:
        """Format LOS client metadata section for prompt"""
        biomaterials = metadata.get("biomaterials") or []
        tube_types = metadata.get("tube_types") or []
        if not biomaterials and not tube_types:
            return ""

        lines = ["Учти при подборе рекомендаций:"]
        lines.extend(f"- Биоматериал: {bm}" for bm in biomaterials)
        lines.extend(f"- Тип пробирки: {tube}" for tube in tube_types)
        return "\n".join(lines)

    @activity.defn(name="build_checkup_interpretation_text")
    async def build_checkup_interpretation_text(self, interpretation_result: Dict) -> str:
        """Extract interpretation summary or introduction"""
        summary = (interpretation_result.get("interpretation_summary") or "").strip()
        if summary:
            return summary
        return (interpretation_result.get("introduction") or "").strip()

    @activity.defn(name="build_checkup_dialog")
    async def build_checkup_dialog(self, props: BuildCheckupDialogProps) -> List[Dict[str, str]]:
        """Build dialog for checkup assistant"""
        LOS_CLIENT_INITIATOR: InitiatorType = "Los_client"

        checkup_prompts = await self.get_checkup_prompts(initiator=props.initiator)

        parts = [
            checkup_prompts.get("preinstruction_text", "").strip(),
            await self.build_checkup_interpretation_text(props.interpretation_result),
        ]

        if props.initiator == LOS_CLIENT_INITIATOR and props.recognition_result is not None:
            metadata = await self.extract_los_client_metadata(props.recognition_result)
            metadata_section = await self.format_los_client_metadata_section(metadata)
            if metadata_section:
                parts.append(metadata_section)

        postinstruction_text = checkup_prompts.get("postinstruction_text", "").strip()
        if postinstruction_text:
            parts.append(postinstruction_text)

        content = "\n\n".join(part for part in parts if part)

        return [{"role": "user", "content": content}]

    @activity.defn(name="convert_dialog_to_chat_messages")
    async def convert_dialog_to_chat_messages(
        self, dialog: List[Dict[str, str]]
    ) -> List[ChatMessage]:
        """Convert dialog format to ChatMessage objects"""
        messages = []
        for msg in dialog:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
        return messages
