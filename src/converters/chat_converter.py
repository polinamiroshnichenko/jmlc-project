import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4


from src import utils
from src.schemas.agents.catalog_assistant import (
    CatalogAssistantResponse,
    CatalogSearchResult,
)

from src.schemas.agents.general_assistant import GeneralAssistantResponse

from src.schemas.agents.checkup_assistant import (
    CheckupAssistantRecommendation,
    CheckupAssistantResponse,
)
from src.schemas.agents.prescription_recognizer import (
    PrescriptionRecognitionNormalizedResult,
)
from src.schemas.agents.recommendation_editor import RecommendationEditorResponse
from src.schemas.chat import (
    ActionContent,
    AgentType,
    ChatFinishedTask,
    Code,
    CodedConcept,
    Coding,
    InitiatorType,
    Report,
    ReportBlock,
    ReportBlockItem,
    ReportContent,
    StructuredOutput,
    TextContent,
    Topic,
    TopicMessage,
)
from src.schemas.flow_inputs.interpretation_with_recomendations import (
    InterpretationWithRecomendationsProps,
)

logger = logging.getLogger(__name__)

_TYPE_LABELS: Dict[str, str] = {
    "lab_analysis": "Laboratory",
    "instrumental_analysis": "Instrumental",
    "doctor": "Consultation",
}

_BLOCK_NAMES: Dict[str, str] = {
    "lab_analysis": "Подобранные анализы",
    "instrumental_analysis": "Инструментальные исследования",
    "doctor": "Врачи",
}

# Названия блоков рекомендаций для dict-ориентированного пути интерпретации
# (включает резервную категорию "other" для нераспознанных типов).
_RECOMMENDATION_BLOCK_NAMES: Dict[str, str] = {
    **_BLOCK_NAMES,
    "other": "Рекомендации",
}


def _convert_codes(codes: List[Code], use_display_name: bool = True) -> List[Coding]:
    return [
        Coding(
            code_system_url=c.codesystem,
            code=c.code,
            display=c.display_name if use_display_name else c.name,
        )
        for c in codes
    ]


def _make_topic_message(
    content: Union[ActionContent, ReportContent, TextContent],
    model: str,
) -> TopicMessage:
    return TopicMessage(
        id=str(uuid4()),
        role="Assistant",
        content=content,
        created_at=datetime.now(timezone.utc),
        model=model,
    )


Recommendation = Union[CheckupAssistantRecommendation, CatalogSearchResult, Code]


def _convert_single_recommendation(r: Recommendation) -> ReportBlockItem:
    rec_type = _TYPE_LABELS.get(r.type, r.type)

    if isinstance(r, Code):
        return ReportBlockItem(
            title=r.name,
            type=rec_type,
            code=CodedConcept(codes=_convert_codes([r], use_display_name=False)),
        )

    codes = r.hxids if isinstance(r, CheckupAssistantRecommendation) else [r.code]
    return ReportBlockItem(
        title=r.name,
        type=rec_type,
        code=CodedConcept(codes=_convert_codes(codes)),
        description=(r.reason if isinstance(r, CheckupAssistantRecommendation) else None),
        triage=(r.triage if isinstance(r, CheckupAssistantRecommendation) else None),
        urgency=(r.urgency if isinstance(r, CheckupAssistantRecommendation) else None),
    )


def _group_and_build_blocks(
    recommendations: List[Recommendation],
) -> List[ReportBlock]:
    by_type: Dict[str, List[Recommendation]] = {k: [] for k in _BLOCK_NAMES}
    for r in recommendations:
        if r.type in by_type:
            by_type[r.type].append(r)

    blocks: List[ReportBlock] = []
    for key, recs in by_type.items():
        if not recs:
            continue
        items = [_convert_single_recommendation(r) for r in recs]
        if key == "doctor":
            for i, item in enumerate(items):
                if item.title == "Терапевт":
                    items.insert(0, items.pop(i))
                    break
        blocks.append(
            ReportBlock(
                title=_BLOCK_NAMES[key],
                items=items,
                type="Recommendation",
            )
        )
    return blocks


def _append_recommendation_response(
    topic: Topic,
    message: str,
    recommendations: Optional[List[Union[CheckupAssistantRecommendation, CatalogSearchResult]]],
    model: str,
    is_finished: bool = False,
) -> Topic:
    if recommendations:
        blocks = _group_and_build_blocks(recommendations)
        content: Union[ReportContent, TextContent] = ReportContent(
            text=message, report=Report(blocks=blocks)
        )
    else:
        content = TextContent(text=message)

    topic.history.append(_make_topic_message(content=content, model=model))

    if is_finished:
        topic.state = "Resolved"

    return topic


def _get_analysis_word(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "анализ"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "анализа"
    return "анализов"


# ── Converters (same order as AGENTS registry) ──────────────────────────────


def convert_general_response(topic: Topic, response: GeneralAssistantResponse) -> Topic:
    redirect = response.redirect_to
    if redirect == AgentType.general_assistant:
        if response.codes:
            blocks = _group_and_build_blocks(response.codes)
            content: Union[ReportContent, TextContent, ActionContent] = ReportContent(
                text=response.message, report=Report(blocks=blocks)
            )
        else:
            content = TextContent(text=response.message)
    else:
        content = ActionContent(
            action=redirect.to_action(),
            text=response.message,
        )

    topic.history.append(_make_topic_message(content=content, model=redirect.value))
    return topic


def convert_checkup_response(topic: Topic, response: CheckupAssistantResponse) -> Topic:
    return _append_recommendation_response(
        topic,
        response.message,
        response.recommendations or None,
        model="checkup-assistant",
        is_finished=response.is_finished,
    )


def convert_catalog_response(topic: Topic, response: CatalogAssistantResponse) -> Topic:
    return _append_recommendation_response(
        topic,
        response.message,
        response.catalog_search_result or None,
        model="catalog-assistant",
        is_finished=response.is_finished,
    )


def convert_prescription_response(
    topic: Topic, response: PrescriptionRecognitionNormalizedResult
) -> Topic:
    items: List[ReportBlockItem] = []
    for r in response.items:
        rec_type = _TYPE_LABELS.get(r.type, r.type) if r.type else None
        coded_concept = None
        if r.code and r.codesystem:
            coded_concept = CodedConcept(
                codes=[
                    Coding(
                        code_system_url=r.codesystem,
                        code=r.code,
                        display=r.display_name,
                    )
                ]
            )
        items.append(ReportBlockItem(title=r.name, type=rec_type, code=coded_concept))

    report = Report(
        blocks=(
            [
                ReportBlock(
                    title="Распознанные анализы",
                    items=items,
                    type="Recommendation",
                )
            ]
            if items
            else []
        )
    )

    if not items:
        message = "Не удалось подобрать анализы по фото."
    else:
        count = len(items)
        message = (
            f"Я подобрал {count} {_get_analysis_word(count)} по вашему фото, "
            "но рекомендуем проверить список. "
            "Возможно, некоторые исследования стоит уточнить или добавить. "
            "Проверьте, все ли анализы соответствуют вашему запросу!"
        )

    topic.history.append(
        _make_topic_message(
            content=ReportContent(text=message, report=report),
            model="prescription-recognizer",
        )
    )
    topic.state = "Resolved"
    return topic


# ── Интерпретация результатов анализов ──────────────────────────────────────
# Этот пласт логики работает с dict-результатами temporal-активностей
# (интерпретация и checkup-рекомендации) и возвращает ChatFinishedTask.


def _convert_hxid_codes(hxids: List[dict]) -> List[Coding]:
    """Конвертирует список hxid-словарей в список Coding."""
    codes: List[Coding] = []
    for hxid in hxids:
        codesystem = hxid.get("codesystem", "hxid")
        if not codesystem.startswith("http"):
            code_system_url = utils.SYSTEM_TO_URL.get(codesystem, codesystem)
        else:
            code_system_url = codesystem

        codes.append(
            Coding(
                code_system_url=code_system_url,
                code=hxid.get("code", ""),
                display=hxid.get("display_name", hxid.get("name", "")),
            )
        )
    return codes


def _convert_checkup_recommendation(
    rec_data: dict,
    reassigned: Optional[str] = None,
) -> ReportBlockItem:
    """Конвертирует dict рекомендации checkup-ассистента в ReportBlockItem."""
    rec_name = rec_data.get("name", "")
    rec_type = rec_data.get("type", "")
    rec_reason = rec_data.get("reason", "")
    rec_urgency = rec_data.get("urgency", "")

    recommendation_type = _TYPE_LABELS.get(rec_type)

    coded_concept = None
    hxids = rec_data.get("hxids", [])
    if hxids:
        codes = _convert_hxid_codes(hxids)
        if codes:
            coded_concept = CodedConcept(codes=codes)

    rec_triage = rec_data.get("triage")
    triage = rec_triage if rec_triage in ("recommended", "optional") else None

    return ReportBlockItem(
        title=rec_name,
        type=recommendation_type,
        description=rec_reason or None,
        code=coded_concept,
        triage=triage,
        urgency=rec_urgency or None,
        reassigned=reassigned,
    )


def _build_obs_index(structured_output: Optional[dict]) -> dict:
    """Строит индекс FHIR-наблюдений из structured_output.

    Возвращает пустой dict, если structured_output не dict или не содержит
    ресурсов.
    """
    if not isinstance(structured_output, dict):
        return {}
    fhir_resources = structured_output.get("resources")
    input_resources = structured_output.get("input_resources")
    return utils.build_observation_index(fhir_resources or input_resources)


def _build_observation_blocks(
    tables: List[dict],
    obs_index: dict,
    initiator: Optional[InitiatorType] = None,
) -> List[ReportBlock]:
    """Строит список блоков ObservationInterpretation из таблиц интерпретации.

    Для каждой строки пытается обогатить данные из FHIR-индекса (obs_index),
    подставляя id FHIR-ресурса и кодировки наблюдения.
    """
    blocks: List[ReportBlock] = []
    for table in tables:
        rows = table.get("rows", [])
        if not rows:
            continue
        table_name = table.get("name", "")
        table_date = utils.format_fhir_date(
            table.get("result_date", "") or table.get("request_date", "")
        )
        items: List[ReportBlockItem] = []
        for row in rows:
            labname = row.get("labname", "")
            if not labname:
                continue

            row_id = row.get("id")
            coded_concept = None

            obs_info = None
            # 1. Поиск по row_id, если он задан
            if row_id:
                obs_info = next(
                    (
                        info
                        for info in obs_index.values()
                        if isinstance(info, dict) and info.get("id") == row_id
                    ),
                    None,
                )

            # 2. Точное совпадение по (table_name, labname, table_date)
            if not obs_info:
                obs_info = obs_index.get((table_name, labname, table_date))

            # 3. Резервное совпадение по (table_name, labname)
            if not obs_info:
                matches = [
                    (date, info)
                    for (t_name, l_name, date), info in obs_index.items()
                    if t_name == table_name and l_name == labname
                ]
                if len(matches) == 1:
                    obs_info = matches[0][1]
                elif len(matches) > 1:
                    obs_info = matches[0][1]
                    logger.warning(
                        f"Ambiguous Observation for table={table_name}, "
                        f"labname={labname}. "
                        f"Multiple dates found: {[m[0] for m in matches]}"
                    )

            if obs_info:
                row_id = obs_info["id"]
                codings = [
                    Coding(
                        code_system_url=c.get("system", ""),
                        code=c.get("code", ""),
                        display=c.get("display"),
                    )
                    for c in obs_info["coding"]
                    if c.get("system") and c.get("code")
                ]
                if codings:
                    coded_concept = CodedConcept(codes=codings)

            row_interpretation = row.get("interpretation") or ""
            row_description = row.get("description") or ""

            if initiator == "Los_client":
                if not row_interpretation:
                    continue
                description = row_interpretation
            else:
                parts = [p for p in (row_description, row_interpretation) if p]
                description = "\n".join(parts).strip()

            items.append(
                ReportBlockItem(
                    id=row_id,
                    title=labname,
                    description=description if description else None,
                    type="Laboratory",
                    code=coded_concept,
                )
            )
        if items:
            blocks.append(
                ReportBlock(
                    title=table_name,
                    items=items,
                    type="ObservationInterpretation",
                )
            )
    return blocks


def _build_recommendation_blocks(
    checkup_recommendations: Optional[dict],
    use_reassigned: bool = False,
) -> List[ReportBlock]:
    """Строит список блоков Recommendation из данных checkup-ассистента.

    Args:
        checkup_recommendations: dict с ключом «recommendations» (список items).
        use_reassigned: если True, поле «reassigned» берётся из данных каждой
            рекомендации (используется для Los_client); иначе передаётся None.
    """
    recommendations_by_type: Dict[str, List[ReportBlockItem]] = {
        key: [] for key in _RECOMMENDATION_BLOCK_NAMES
    }

    if checkup_recommendations and isinstance(checkup_recommendations, dict):
        recommendations_data = checkup_recommendations.get("recommendations", [])
        for rec_data in recommendations_data:
            if not isinstance(rec_data, dict):
                continue
            rec_type = rec_data.get("type")
            reassigned = rec_data.get("reassigned") if use_reassigned else None
            item = _convert_checkup_recommendation(rec_data, reassigned=reassigned)
            if rec_type in recommendations_by_type:
                recommendations_by_type[rec_type].append(item)
            else:
                recommendations_by_type["other"].append(item)

    blocks: List[ReportBlock] = []
    for type_key in ("lab_analysis", "instrumental_analysis", "doctor", "other"):
        items = recommendations_by_type[type_key]
        if not items:
            continue
        blocks.append(
            ReportBlock(
                title=_RECOMMENDATION_BLOCK_NAMES[type_key],
                items=items,
                type="Recommendation",
            )
        )
    return blocks


def _format_report_block(block: ReportBlock) -> str:
    """Форматирует один блок отчёта с его рекомендациями."""
    if not block.items:
        return block.title or ""

    recs: List[str] = []
    for i, rec in enumerate(block.items, start=1):
        item = f"{i}) {rec.title}"
        if rec.description:
            item += f"\n{rec.description}"
        if rec.type == "Laboratory" and rec.code:
            item += "\nСоответствующие анализы в нашем каталоге:"
            for code in rec.code.codes:
                item += f"\n{code.display}"
        recs.append(item)

    recs_str = "\n".join(recs).strip()
    return f"{block.title}:\n{recs_str}"


def _format_report(report: Optional[Report]) -> Optional[str]:
    """Форматирует блоки отчёта в читаемую строку (или None, если пусто)."""
    if not report or not report.blocks:
        return None

    sections: List[str] = []
    for block in report.blocks:
        formatted_block = _format_report_block(block)
        if formatted_block:
            sections.append(formatted_block)

    if not sections:
        return None

    return "\n\n".join(sections).strip()


def _add_report_message_to_topic(
    topic: Topic,
    text: str,
    report: Optional[Report],
) -> Topic:
    """Добавляет новое сообщение Assistant с отчётом в историю топика."""
    new_message = TopicMessage(
        id=str(uuid4()),
        role="Assistant",
        content=ReportContent(
            text=text,
            report=report,
            type="Report",
        ),
        created_at=datetime.now(timezone.utc),
        model=topic.history[-1].model if topic.history else None,
    )
    history = topic.history.copy()
    history.append(new_message)
    topic.history = history
    return topic


def extract_action_type(topic: Topic) -> Optional[str]:
    if not topic.history:
        return None
    content = topic.history[-1].content
    return getattr(content, "action", None)


def build_context_from_topic(topic: Topic) -> str:
    """Собирает текстовый контекст диалога из истории топика.

    Объединяет текстовые сообщения (TextContent / ActionContent) и форматированные
    отчёты (ReportContent) с префиксами ролей. Используется при формировании
    контекста для интерпретации (см. InterpretationWithRecomendationsFlow).
    """
    role_prefixes = {
        "User": "Пользователь",
        "Assistant": "Агент",
        "Service": "Агент",
    }

    context_parts: List[str] = []
    for m in topic.history or []:
        message_text = ""

        if isinstance(m.content, TextContent):
            message_text = m.content.text or ""
        elif isinstance(m.content, ActionContent):
            message_text = m.content.text or ""
        elif isinstance(m.content, ReportContent):
            if (m.model or "") not in (
                "interpretation-assistant",
                "interpretation-with-recommendations-assistant",
            ):
                formatted_report = _format_report(m.content.report)
                if formatted_report:
                    message_text = (m.content.text + "\n" + formatted_report).strip()

        if message_text:
            context_parts.append(f"{role_prefixes.get(m.role, '')}: {message_text}")

    return "\n".join(context_parts)


def convert_request(request: Topic) -> InterpretationWithRecomendationsProps:
    # Импорт отложен: src.schemas.cds тянет ассеты деплоя (src.constants),
    # которые недоступны при обычном импорте модуля.
    from src.schemas.cds import DocumentReference, RecognitionSubmission

    kind_map = {
        "RecognizePrescription": "prescription",
        "ExplainLabtestResult": "interpretation",
        "InterpretateWithRecommendations": "interpretation_with_recommendations",
    }
    action_type = extract_action_type(request)
    kind = kind_map.get(action_type, action_type)

    messages = request.history or []

    references: List[DocumentReference] = []
    context_parts: List[str] = []
    file_index = 1

    role_prefixes = {
        "User": "Пользователь",
        "Assistant": "Агент",
        "Service": "Агент",
    }

    for m in messages:
        message_text = ""

        if isinstance(m.content, TextContent):
            message_text = m.content.text or ""
        elif isinstance(m.content, ActionContent):
            message_text = m.content.text or ""
        elif isinstance(m.content, ReportContent):
            if (m.model or "") not in (
                "interpretation-assistant",
                "interpretation-with-recommendations-assistant",
            ):
                formatted_report = _format_report(m.content.report)
                if formatted_report:
                    message_text = (m.content.text + "\n" + formatted_report).strip()

        if message_text:
            context_parts.append(f"{role_prefixes.get(m.role, '')}: {message_text}")

        for a in (m.attachment or []) or []:
            if not a or not a.link or not a.mime_type:
                continue
            references.append(
                DocumentReference(
                    url=a.link,
                    mime_type=a.mime_type,
                    name=f"file_{file_index}.{a.mime_type.split('/')[-1]}",
                )
            )
            file_index += 1

    return RecognitionSubmission(
        kind=kind,
        references=references,
        callback_url=request.service_attributes.get("callback_url"),
        context="\n".join(context_parts),
    )


def _build_interpretation_report(
    interpretation_data: dict,
    extra_blocks: Optional[List[ReportBlock]] = None,
    initiator: Optional[InitiatorType] = None,
) -> Tuple[Report, str]:
    """Строит Report и текст сообщения из dict результата интерпретации.

    Returns:
        (report, text), где report — экземпляр Report, а text — строка-резюме
        интерпретации (может быть пустой).
    """
    structured_output = interpretation_data.get("structured_output") or {}
    obs_index = _build_obs_index(structured_output)
    observation_blocks = _build_observation_blocks(
        interpretation_data.get("tables", []), obs_index, initiator=initiator
    )
    blocks = (extra_blocks or []) + observation_blocks

    structured_output_data = {
        "custom_object": {
            "shortSummary": interpretation_data.get(
                "introduction", "Не удалось интерпретировать результаты анализов."
            ),
        },
        "resources": structured_output.get("resources") or None,
    }

    report = Report(
        blocks=blocks if blocks else None,
        structured_output=StructuredOutput(**structured_output_data),
    )
    text = interpretation_data.get("interpretation_summary", "")
    return report, text


def convert_interpretation_response(result: dict, request: Topic) -> ChatFinishedTask:
    report, interpretation_text = _build_interpretation_report(
        result["result"], initiator=request.initiator
    )

    request.state = "Resolved"
    result_topic = _add_report_message_to_topic(
        topic=request,
        text=interpretation_text,
        report=report,
    )

    return ChatFinishedTask(
        taskId=result["task_id"],
        status=result["status"],
        result=result_topic,
    )


def convert_interpretation_with_recommendations_response(
    result: dict, request: Topic
) -> ChatFinishedTask:
    """Конвертирует результат интерпретации с checkup в ChatFinishedTask."""
    result_data = result["result"]
    interpretation = result_data.get("interpretation", {})
    checkup_recommendations = result_data.get("checkup_recommendations")

    is_los = getattr(request, "initiator", None) == "Los_client"

    recommendation_blocks = _build_recommendation_blocks(
        checkup_recommendations, use_reassigned=is_los
    )
    report, interpretation_text = _build_interpretation_report(
        interpretation,
        extra_blocks=recommendation_blocks,
        initiator=request.initiator,
    )

    final_text = interpretation_text or "Результаты интерпретации анализов."

    request.state = "Resolved"
    result_topic = _add_report_message_to_topic(
        topic=request,
        text=final_text,
        report=report,
    )

    return ChatFinishedTask(
        taskId=result["task_id"],
        status=result["status"],
        result=result_topic,
    )


def convert_interpretation_with_recommendations_callback(topic: Topic, response) -> Topic:
    """Конвертер для send_callback: (topic, response) -> Topic.

    В отличие от convert_interpretation_with_recommendations_response (dict -> ChatFinishedTask,
    используется для Temporal-выхода), эта функция совместима с сигнатурой send_callback и
    принимает модель InterpretationWithRecommendationsResult.
    """
    if hasattr(response, "model_dump"):
        result_data = response.model_dump(by_alias=True)
    else:
        result_data = response or {}

    interpretation = result_data.get("interpretation", {}) or {}
    checkup_recommendations = result_data.get("checkup_recommendations")

    is_los = getattr(topic, "initiator", None) == "Los_client"

    recommendation_blocks = _build_recommendation_blocks(
        checkup_recommendations, use_reassigned=is_los
    )
    report, interpretation_text = _build_interpretation_report(
        interpretation,
        extra_blocks=recommendation_blocks,
        initiator=topic.initiator,
    )

    final_text = interpretation_text or "Результаты интерпретации анализов."

    topic.state = "Resolved"
    return _add_report_message_to_topic(
        topic=topic,
        text=final_text,
        report=report,
    )


def convert_recommendation_editor_response(
    topic: Topic, response: RecommendationEditorResponse
) -> Topic:
    topic.history.append(
        _make_topic_message(
            content=TextContent(text=response.text),
            model="recommendation-editor",
        )
    )
    return topic


def convert_recommendation_enricher_response(
    topic: Topic, response: RecommendationEditorResponse
) -> Topic:
    topic.history.append(
        _make_topic_message(
            content=TextContent(text=response.text),
            model="recommendation-enricher",
        )
    )
    return topic
