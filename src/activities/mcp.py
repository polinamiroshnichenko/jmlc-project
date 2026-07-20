import uuid
import json

from temporalio import activity
from pydantic import BaseModel

from src.clients.mcp_enrichment import McpEnrichmentService
from src.schemas.recognition import ReportRecognitionResult
from src.activities.recognition_data_processing import parse_date


class GetMcpTables(BaseModel):
    recognition_result: ReportRecognitionResult
    topic_id: str
    patient_info: dict | None


class McpActivities:

    def __init__(self, mcp_service: McpEnrichmentService):
        self.mcp_service = mcp_service

    @activity.defn(name="get_mcp_tables")
    async def __call__(self, data: GetMcpTables) -> str | None:
        mcp_tables = await self.mcp_service(
            data.recognition_result, data.topic_id, data.patient_info
        )
        if mcp_tables:
            historical_data = [table.model_dump() for table in mcp_tables]
            for table in historical_data:
                for row in table.get("rows", []):
                    row["id"] = str(uuid.uuid4())
            historical_data = sorted(historical_data, key=parse_date, reverse=False)
            activity.logger.info(f"MCP enrichment: added{len(mcp_tables)} historical tables")
            return json.dumps(historical_data, ensure_ascii=False, indent=2)
        return
