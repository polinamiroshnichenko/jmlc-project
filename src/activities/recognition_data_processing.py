import uuid
from datetime import datetime

from pydantic import BaseModel
from temporalio import activity

from src.schemas.recognition import RecognizedTable, ReportInterpretatedResult


@activity.defn
async def preprocess_recognition_data(data: list[RecognizedTable]) -> list[dict]:
    data = [table.model_dump() for table in data]
    for table in data:
        for row in table.get("rows", []):
            row["id"] = row.get("observation_id") or str(uuid.uuid4())
    return sorted(data, key=parse_date, reverse=False)


def parse_date(entry: dict):
    date_str = entry.get("request_date") or entry.get("result_date")
    if not date_str:
        return datetime.max
    try:
        dt = datetime.fromisoformat(str(date_str))
        return dt.replace(tzinfo=None)
    except Exception:
        return datetime.max


class PostprocessInterpretationProps(BaseModel):
    raw_data: list[dict]
    interpreted_data: ReportInterpretatedResult


@activity.defn
async def postprocess_interpretation(props: PostprocessInterpretationProps) -> dict:
    interpreted_index = {
        row["id"]: row
        for table in props.interpreted_data.model_dump()["tables"]
        for row in table["rows"]
    }

    tables = []

    for table in props.raw_data:
        rows = []
        for row in table["rows"]:
            merged_row = row.copy()
            if interpreted := interpreted_index.get(row["id"]):
                merged_row.update(
                    description=interpreted["description"],
                    interpretation=interpreted["interpretation"],
                    is_norm=interpreted["is_norm"],
                )
            rows.append(merged_row)
        tables.append(
            {
                **table,
                "rows": rows,
            }
        )

    return {
        "tables": tables,
        "introduction": props.interpreted_data.introduction,
        "interpretation_summary": props.interpreted_data.interpretation_summary,
    }
