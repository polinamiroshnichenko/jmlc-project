import logging
import uuid
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

from dateutil import parser
from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.composition import Composition, CompositionSection
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.extension import Extension
from fhir.resources.identifier import Identifier
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient
from fhir.resources.period import Period
from fhir.resources.quantity import Quantity
from fhir.resources.reference import Reference
from fhir.resources.specimen import Specimen

from src.ports.assets_repository import AssetsRepository
from src.schemas.recognition import ReportNormalizedResult, ReportRecognitionResult

logger = logging.getLogger(__name__)


class FhirToReportResultConverter:
    """Converts FHIR Bundle format to internal recognition format."""

    def __call__(self, document: Dict) -> List[Dict]:
        """Convert FHIR document to internal format."""
        self._index_bundle(document)
        return ReportRecognitionResult(items=self._transform_entries())

    def _index_bundle(self, document: Dict) -> None:
        self.entries = document.get("entry", [])
        self._specimens_by_id: Dict[str, Dict] = {}

        for entry in self.entries:
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType")
            if rtype == "Specimen":
                sid = resource.get("id")
                if sid:
                    self._specimens_by_id[sid] = resource

    def _transform_entries(self) -> List[Dict]:
        """Transform FHIR entries to internal format."""
        patient_info = self._extract_patient()
        laboratory = self._extract_lab_name()
        reports = self._extract_reports()

        for report in reports:
            report.update(patient_info)
            report["laboratory"] = laboratory

        return reports

    def _extract_patient(self) -> Dict:
        """Extract patient information from FHIR Patient resource."""
        for entry in self.entries:
            if entry.get("resource", {}).get("resourceType") == "Patient":
                patient = entry["resource"]
                return {
                    "date_of_birth": patient.get("birthDate", ""),
                    "gender": patient.get("gender", ""),
                }
        return {"date_of_birth": "", "gender": ""}

    def _extract_lab_name(self) -> str:
        """Extract laboratory name from FHIR Composition resource."""
        for entry in self.entries:
            if entry.get("resource", {}).get("resourceType") == "Composition":
                author = entry["resource"].get("author", [{}])[0]
                return author.get("display", "")
        return ""

    def _extract_specimen_by_reference(self, ref_id: str) -> Optional[Dict]:
        return self._specimens_by_id.get(ref_id)

    def _parse_reference_id(self, reference: str) -> str:
        if ":" in reference:
            return reference.split(":")[-1]
        return reference.split("/")[-1]

    def _extract_container_type(self, specimen: Dict) -> List[str]:
        """Extract all container types."""
        if not specimen:
            return []
        container_types = []
        for container in specimen.get("container") or []:
            if not isinstance(container, dict):
                continue
            container_type = container.get("type") or {}
            for coding in container_type.get("coding") or []:
                code = (coding.get("code") or "").strip()
                if code:
                    container_types.append(code)

        return list(dict.fromkeys(container_types))

    def _extract_biomaterial(self, specimen: Dict) -> List[str]:
        """Extract all biomaterial types."""
        biomaterials = []
        codings = specimen.get("type", {}).get("coding", [])
        for coding in codings:
            if "display" in coding:
                biomaterials.append(coding["display"])
            elif "code" in coding:
                biomaterials.append(coding["code"])
        return list(dict.fromkeys(biomaterials))

    def _extract_unique_specimens_from_bundle(self, document: Dict) -> List[Dict]:
        seen = set()
        result = []

        def add(spec):
            sid = spec.get("id") or id(spec)
            if sid in seen:
                return
            seen.add(sid)
            result.append(spec)

        for spec in self._specimens_by_id.values():
            add(spec)

        for entry in self.entries:
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "DiagnosticReport":
                for spec_ref in resource.get("specimen", []):
                    ref = spec_ref.get("reference", "")
                    if not ref:
                        continue
                    sid = self._parse_reference_id(ref)
                    spec = self._specimens_by_id.get(sid)
                    if spec:
                        add(spec)

        return result

    def extract_report_hxids_from_bundle(self, document: Dict) -> List[str]:
        self._index_bundle(document)
        hxids = []
        for entry in self.entries:
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "DiagnosticReport":
                extensions = resource.get("extension", [])
                for ext in extensions:
                    if not isinstance(ext, dict):
                        continue
                    url = ext.get("url", "")
                    if "nomenclature" in url:
                        hxid_code = ext.get("valueString", "")
                        if hxid_code:
                            hxids.append(hxid_code.strip())

        return list(dict.fromkeys(hxids))

    def extract_tube_types_from_bundle(self, document: Dict) -> List[str]:
        """Extract tube types."""
        self._index_bundle(document)
        tube_types = []
        for specimen in self._specimens_by_id.values():
            tube_types.extend(self._extract_container_type(specimen))

        return list(dict.fromkeys(tube_types))

    def extract_specimen_container_pairs_from_bundle(self, document: Dict) -> List[Tuple[str, str]]:
        """Extract (biomaterial, tube_type) pairs."""
        self._index_bundle(document)
        specimens = self._extract_unique_specimens_from_bundle(document)

        pairs = []
        for specimen in specimens:
            biomaterials = self._extract_biomaterial(specimen)
            tube_types = self._extract_container_type(specimen)

            for biomaterial in biomaterials:
                for tube_type in tube_types:
                    if biomaterial and tube_type:
                        pairs.append((biomaterial, tube_type))

        return list(dict.fromkeys(pairs))

    def _format_reference_range(self, ref_range: List[Dict]) -> str:
        """Format reference range from FHIR format to string."""
        if not ref_range:
            return ""

        range_item = ref_range[0]
        low = range_item.get("low", {})
        high = range_item.get("high", {})

        low_val = low.get("value")
        high_val = high.get("value")
        low_comp = low.get("comparator")
        high_comp = high.get("comparator")

        if low_val is not None and high_val is not None:
            return f"{low_val} - {high_val}"
        elif high_val is not None:
            comparator = high_comp or "<="
            return f"{comparator} {high_val}"
        elif low_val is not None:
            comparator = low_comp or ">="
            return f"{comparator} {low_val}"
        elif "text" in range_item:
            return range_item["text"]
        else:
            return ""

    def _extract_measure(self, value_quantity: Dict) -> str:
        """Extract measurement unit from value quantity."""
        extension = value_quantity.get("extension", [{}])
        for item in extension:
            if "valueCoding" in item:
                return item.get("valueCoding", {}).get("display", "")
        return ""

    def _extract_reports(self) -> List[Dict]:
        """Extract diagnostic reports and their observations."""
        results = []
        for entry in self.entries:
            res = entry.get("resource", {})
            if res.get("resourceType") == "DiagnosticReport":
                issued_date = res.get("issued", "")
                report = {
                    "result_date": (
                        parser.isoparse(issued_date).date().strftime("%Y-%m-%d")
                        if issued_date
                        else None
                    ),
                    "request_date": (
                        parser.isoparse(issued_date).date().strftime("%Y-%m-%d")
                        if issued_date
                        else None
                    ),
                    "name": res.get("code", {}).get("coding", [{}])[0].get("display", ""),
                    "biomaterial": "",
                    "rows": [],
                    "comment": res.get("conclusion", ""),
                }

                # Extract biomaterial from specimen
                specimens = res.get("specimen", [])
                for specimen_ref in specimens:
                    ref = specimen_ref.get("reference", "")
                    ref_id = ref.split(":")[-1] if ":" in ref else ref.split("/")[-1]
                    if ref_id:
                        specimen = self._extract_specimen_by_reference(ref_id)
                        if specimen:
                            biomaterials = self._extract_biomaterial(specimen)
                            if biomaterials:
                                report["biomaterial"] = biomaterials[0]
                            break

                # Extract observations
                result_refs = []
                for r in res.get("result", []):
                    ref = r.get("reference", "")
                    ref_id = ref.split(":")[-1] if ":" in ref else ref.split("/")[-1]
                    result_refs.append(ref_id)

                for ref in result_refs:
                    for obs_entry in self.entries:
                        obs_res = obs_entry.get("resource", {})
                        if (
                            obs_res.get("resourceType") == "Observation"
                            and obs_res.get("id") == ref
                        ):
                            code = obs_res.get("code", {})
                            coding = code.get("coding", [{}])
                            value = obs_res.get("valueQuantity", {}).get("value") or obs_res.get(
                                "valueString", ""
                            )
                            measure = self._extract_measure(obs_res.get("valueQuantity", {}))
                            ref_range = obs_res.get("referenceRange", [])
                            ref_val = self._format_reference_range(ref_range)

                            report["rows"].append(
                                {
                                    "labname": code.get("text") or coding[0].get("display", ""),
                                    "result": str(value),
                                    "measure": measure,
                                    "ref_value": ref_val,
                                    "comment": "",
                                    "observation_id": obs_res.get("id"),
                                }
                            )

                results.append(report)

        return results


class ReportResultToFhirConverter:
    """Convert internal recognition format to FHIR Bundle (typed resources)."""

    def __init__(self, assets: AssetsRepository):
        self._assets = assets

    def _iso_date(self, date_str: Optional[str]) -> Optional[str]:
        """Convert date string to ISO datetime format. Returns None if date is empty."""
        if not date_str or not date_str.strip():
            return None
        try:
            d = date.fromisoformat(date_str.strip())
            return f"{d.isoformat()}T00:00:00+00:00"
        except ValueError:
            return None

    def _iso_datetime_now(self) -> str:
        """Get current datetime in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _get_period_dates(
        self, request_date: Optional[str], result_date: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine period boundaries for sampling/result dates.
        - When both dates are known, keep them as-is.
        - When only one is known, mirror it to the missing value.
        - When neither is known, both values stay empty.
        """
        if request_date and result_date:
            return request_date, result_date
        if request_date:
            return request_date, request_date
        if result_date:
            return result_date, result_date
        return None, None

    def _safe_coding(
        self, system: Optional[str], code: Optional[str], display: Optional[str]
    ) -> Coding:
        """Create Coding safely — omit empty `code`."""
        kwargs = {}
        if system:
            kwargs["system"] = system
        if display:
            kwargs["display"] = display
        if code and str(code).strip():
            kwargs["code"] = str(code).strip()
        return Coding(**kwargs)

    def convert_to_fhir(self, recognition_result: Dict) -> List[Bundle]:
        """Main converter - returns list of Bundles."""
        if not recognition_result or not recognition_result.get("tables"):
            return []

        tables = recognition_result.get("tables", [])

        # Group tables by laboratory and request_date
        grouped_tables = self._group_tables_by_order(tables)

        # Create individual bundles for each group and return as list
        bundles: List[Bundle] = []
        for order_key, order_tables in grouped_tables.items():
            order_bundle = self._create_bundle_for_order(order_tables)
            if order_bundle:
                bundles.append(order_bundle)

        return bundles

    def _group_tables_by_order(self, tables: List[Dict]) -> Dict[tuple, List[Dict]]:
        """Group tables by order (laboratory + request_date)."""
        grouped = {}
        for table in tables:
            lab = table.get("laboratory", "").strip()
            req_date = table.get("request_date", "").strip()

            # If both lab and date are empty, put in special group
            if not lab and not req_date:
                order_key = ("_no_info", "_no_info")
            else:
                order_key = (lab or "_no_lab", req_date or "_no_date")

            if order_key not in grouped:
                grouped[order_key] = []
            grouped[order_key].append(table)

        return grouped

    def _group_tables_by_name(self, tables: List[Dict]) -> Dict[str, List[Dict]]:
        """Group tables by diagnostic report name."""
        grouped = {}
        for table in tables:
            name = table.get("name", "").strip()
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(table)
        return grouped

    def _merge_tables_with_same_name(self, tables: List[Dict]) -> Dict:
        """Merge tables with the same name into one table, combining all rows.

        If there are identical labnames with different results, both are kept.
        """
        if not tables:
            return {}

        # Use the first table as base
        merged = dict(tables[0])

        # Combine all rows from all tables (keep all, even duplicates with different results)
        all_rows = []
        for table in tables:
            all_rows.extend(table.get("rows", []))
        merged["rows"] = all_rows

        # Use the earliest request_date and result_date if available
        request_dates = [t.get("request_date", "") for t in tables if t.get("request_date")]
        result_dates = [t.get("result_date", "") for t in tables if t.get("result_date")]

        if request_dates:
            merged["request_date"] = request_dates[0]  # Keep first non-empty
        if result_dates:
            merged["result_date"] = result_dates[0]  # Keep first non-empty

        # Combine comments if any
        comments = [t.get("comment", "") for t in tables if t.get("comment")]
        if comments:
            merged["comment"] = " ".join(filter(None, comments))

        return merged

    def _create_bundle_for_order(self, tables: List[Dict]) -> Optional[Bundle]:
        """Create a FHIR Bundle for one order (group of tables)."""
        tables = self._filter_tables_with_values(tables)
        if not tables:
            return None

        patient_id = str(uuid.uuid4())
        composition_id = str(uuid.uuid4())

        patient = self._create_patient_resource_from_tables(tables, patient_id)
        composition = self._create_composition_resource_from_tables(
            tables, composition_id, patient_id
        )

        # Group tables by name (diagnostic report name)
        tables_by_name = self._group_tables_by_name(tables)

        diagnostic_reports: List[DiagnosticReport] = []
        observations: List[Observation] = []
        specimens: List[Specimen] = []
        diagnostic_report_refs: List[Reference] = []

        for report_name, name_tables in tables_by_name.items():
            # Merge tables with the same name
            merged_table = self._merge_tables_with_same_name(name_tables)

            specimen_id = str(uuid.uuid4())
            specimen = self._create_specimen_resource(merged_table, specimen_id)
            if specimen:
                specimens.append(specimen)
            else:
                specimen_id = ""

            table_observation_ids: List[str] = []
            for row in merged_table.get("rows", []):
                observation_id = str(uuid.uuid4())
                observation = self._create_observation_resource(
                    row,
                    observation_id,
                    patient_id,
                    specimen_id,
                    merged_table.get("request_date", ""),
                    merged_table.get("result_date", ""),
                )
                observations.append(observation)
                table_observation_ids.append(observation_id)

            diagnostic_report_id = str(uuid.uuid4())
            diagnostic_report = self._create_diagnostic_report_resource(
                merged_table,
                diagnostic_report_id,
                patient_id,
                table_observation_ids,
            )
            diagnostic_reports.append(diagnostic_report)
            diagnostic_report_refs.append(Reference(reference=f"urn:uuid:{diagnostic_report_id}"))

        # Composition section
        if not composition.section:
            composition.section = [CompositionSection(title="Diagnostic Reports", entry=[])]
        composition.section[0].entry = diagnostic_report_refs

        entries: List[BundleEntry] = []
        entries.append(
            BundleEntry.construct(fullUrl=f"urn:uuid:{composition_id}", resource=composition)
        )
        entries.append(BundleEntry.construct(fullUrl=f"urn:uuid:{patient_id}", resource=patient))
        for dr in diagnostic_reports:
            entries.append(BundleEntry.construct(fullUrl=f"urn:uuid:{dr.id}", resource=dr))
        for sp in specimens:
            entries.append(BundleEntry.construct(fullUrl=f"urn:uuid:{sp.id}", resource=sp))
        for ob in observations:
            entries.append(BundleEntry.construct(fullUrl=f"urn:uuid:{ob.id}", resource=ob))

        bundle_id = str(uuid.uuid4())

        return Bundle.construct(
            resourceType="Bundle",
            type="document",
            id=bundle_id,
            identifier=[
                Identifier(
                    system="http://ml.example.com/laboratory/external-test",
                    value=bundle_id,
                )
            ],
            timestamp=self._iso_datetime_now(),
            entry=entries,
        )

    def _create_patient_resource_from_tables(
        self, tables: List[Dict], patient_uuid: str
    ) -> Patient:
        """Create patient resource from tables."""
        first_table = tables[0]
        return Patient(
            id=patient_uuid,
            birthDate=first_table.get("date_of_birth"),
            gender=first_table.get("gender"),
        )

    def _create_composition_resource_from_tables(
        self, tables: List[Dict], composition_uuid: str, patient_uuid: str
    ) -> Composition:
        """Create composition resource from tables with proper dates."""
        first_table = tables[0]
        laboratory_name = first_table.get("laboratory", "")

        if not laboratory_name:
            laboratory_name = "Неизвестная лаборатория"

        # Get dates for event period
        request_date = None
        result_date = None
        for table in tables:
            if table.get("request_date"):
                request_date = self._iso_date(table.get("request_date"))
                break
        for table in tables:
            if table.get("result_date"):
                result_date = self._iso_date(table.get("result_date"))
                break

        # Create event with period
        start_date, end_date = self._get_period_dates(request_date, result_date)
        event = None
        if start_date or end_date:
            event = {"period": Period(start=start_date, end=end_date)}

        composition = Composition(
            id=composition_uuid,
            status="final",
            type=CodeableConcept(
                coding=[self._safe_coding(system="http://loinc.org", code="11502-2", display=None)],
                text="Laboratory report",
            ),
            subject=Reference(reference=f"urn:uuid:{patient_uuid}"),
            author=[Reference(display=laboratory_name)],
            date=self._iso_datetime_now(),  # Generation date
            title="Laboratory report",
            section=[CompositionSection(title="Diagnostic Reports", entry=[])],
            event=[event] if event else None,
        )

        return composition

    def _create_specimen_resource(self, table: Dict, specimen_uuid: str) -> Specimen:
        name = table.get("biomaterial")
        if name:
            biomaterials = self._assets.biomaterials()
            if name in biomaterials:
                asset = biomaterials[name]
                coding = asset.coding[0] if asset.coding else None
                return Specimen(
                    id=specimen_uuid,
                    type=CodeableConcept(
                        coding=[
                            self._safe_coding(
                                system=coding.code_system_url if coding else None,
                                code=coding.code if coding else None,
                                display=name,
                            )
                        ],
                        text=name,
                    ),
                )
            # Biomaterial not found in dictionary - create specimen with text only
            logger.warning(
                f"Biomaterial '{name}' not found in BIOMATERIALS dictionary, creating specimen with text only"
            )
            return Specimen(
                id=specimen_uuid,
                type=CodeableConcept(
                    text=name,
                ),
            )
        return None

    def _create_diagnostic_report_resource(
        self,
        table: Dict,
        diagnostic_report_uuid: str,
        patient_uuid: str,
        observation_ids: List[str],
    ) -> DiagnosticReport:
        request_date_iso = self._iso_date(table.get("request_date"))
        result_date_iso = self._iso_date(table.get("result_date"))
        effective, issued = self._get_period_dates(request_date_iso, result_date_iso)
        name = table.get("name")
        laboratory_name = table.get("laboratory", "")

        # nomenclatures = {
        #     value["commercial_name"]: {"hxid": key, "components": value["components"]}
        #     for key, value in utils.NOMENCLATURE.items()
        # }

        if name:
            coding = [
                self._safe_coding(
                    system="http://ml.example.com/laboratory/external-test-code",
                    code=name,
                    display=name,
                )
            ]
        else:
            coding = None

        dr = DiagnosticReport(
            id=diagnostic_report_uuid,
            status="final",
            code=CodeableConcept(
                coding=coding or None,
                text=name or None,
            ),
            subject=Reference(reference=f"urn:uuid:{patient_uuid}"),
            effectiveDateTime=effective,
            issued=issued,
            performer=[Reference(display=laboratory_name)] if laboratory_name else None,
            result=[Reference(reference=f"urn:uuid:{oid}") for oid in observation_ids],
            identifier=[
                Identifier(
                    system="http://ml.example.com/laboratory/external-test-id",
                    value=str(uuid.uuid4()),
                )
            ],
        )

        return dr

    def _create_observation_resource(
        self,
        row: Dict,
        observation_uuid: str,
        patient_uuid: str,
        specimen_uuid: str,
        request_date: str,
        result_date: str,
    ) -> Observation:
        code = row.get("code", "")
        codesystem = row.get("codesystem", "")
        display = row.get("display", "")
        text = row.get("text", "")

        # Build CodeableConcept
        coding_list = []
        if code and codesystem:
            coding_list.append(
                self._safe_coding(
                    system=codesystem,
                    code=code,
                    display=display if display else None,
                )
            )

        # Use text (original recognized value) if available, otherwise use labname
        code_text = text if text else row.get("labname", "Unknown")

        labname_code = CodeableConcept(
            coding=coding_list if coding_list else None,
            text=code_text,
        )

        request_date_iso = self._iso_date(request_date)
        result_date_iso = self._iso_date(result_date)
        effective, issued = self._get_period_dates(request_date_iso, result_date_iso)

        obs = Observation(
            id=observation_uuid,
            status="final",
            category=[
                CodeableConcept(
                    coding=[
                        self._safe_coding(
                            system="http://hl7.org/fhir/observation-category",
                            code="laboratory",
                            display=None,
                        )
                    ]
                )
            ],
            code=labname_code,
            subject=Reference(reference=f"urn:uuid:{patient_uuid}"),
            effectiveDateTime=effective,
            issued=issued,
            identifier=[
                Identifier(
                    system="https://api.example.com/terminology/observation-guid",
                    value=str(uuid.uuid4()),
                )
            ],
        )

        if specimen_uuid:
            obs.specimen = Reference(reference=f"urn:uuid:{specimen_uuid}")

        result_val = row.get("result")
        if self._row_has_value(row):
            try:
                value = float(str(result_val).replace(",", "."))
                measure = row.get("measure")
                qty = Quantity(
                    value=value,
                    unit=measure or None,
                    system="http://unitsofmeasure.org/" if measure else None,
                    code=measure or None,
                )
                mc = row.get("measure_code")
                coding_args = {
                    "system": "https://nsi.rosminzdrav.ru/dictionaries/1.2.643.5.1.13.13.11.1358",
                    "display": measure or None,
                }
                if mc and str(mc).strip():
                    coding_args["code"] = str(mc).strip()
                qty.extension = [
                    Extension(
                        url="https://api.example.com/extensions/nsi-units",
                        valueCoding=Coding(**coding_args),
                    )
                ]
                obs.valueQuantity = qty
            except (ValueError, TypeError):
                obs.valueString = str(result_val)

        # interpretation
        interpretation_code = row.get("interpretation_code", "N")
        if interpretation_code in ["N", "L", "H", "A"]:
            obs.interpretation = [
                CodeableConcept(
                    coding=[
                        self._safe_coding(
                            system="http://hl7.org/fhir/v2/0078",
                            code=interpretation_code,
                            display=None,
                        )
                    ]
                )
            ]
        else:
            obs.interpretation = [
                CodeableConcept(
                    coding=[
                        self._safe_coding(
                            system="http://hl7.org/fhir/v2/0078",
                            code="N",
                            display=None,
                        )
                    ]
                )
            ]

        # referenceRange
        ref_value = row.get("ref_value")
        if ref_value:
            rr_items = []
            measure = row.get("measure", "")
            if isinstance(ref_value, dict):
                low = ref_value.get("low")
                high = ref_value.get("high")
                if low is not None or high is not None:
                    rr = {
                        "type": {
                            "coding": [
                                {
                                    "system": "http://hl7.org/fhir/referencerange-meaning",
                                    "code": "normal",
                                }
                            ]
                        },
                    }
                    if low is not None:
                        rr["low"] = {
                            "value": low,
                            "unit": measure or None,
                            "system": "http://unitsofmeasure.org/",
                            "code": measure or None,
                        }
                    if high is not None:
                        rr["high"] = {
                            "value": high,
                            "unit": measure or None,
                            "system": "http://unitsofmeasure.org/",
                            "code": measure or None,
                        }
                    rr_items.append(rr)
                if ref_value.get("text"):
                    rr_items.append({"text": ref_value["text"]})
            elif isinstance(ref_value, str):
                rr_items.append({"text": ref_value})
            if rr_items:
                obs.referenceRange = rr_items

        if row.get("interpretation"):
            obs.note = [{"text": str(row.get("interpretation"))}]

        obs.extension = obs.extension or []
        show_graph = False
        if obs.valueQuantity:
            show_graph = True
        obs.extension.append(
            Extension(
                url="https://api.example.com/extension/show-graph",
                valueBoolean=show_graph,
            )
        )

        return obs

    def _filter_tables_with_values(self, tables: List[Dict]) -> List[Dict]:
        """Keep only tables that contain rows with usable observation values."""
        filtered_tables: List[Dict] = []
        for table in tables:
            rows = [row for row in table.get("rows", []) if self._row_has_value(row)]
            if not rows:
                continue
            table_copy = dict(table)
            table_copy["rows"] = rows
            filtered_tables.append(table_copy)
        return filtered_tables

    def _row_has_value(self, row: Dict) -> bool:
        """Check whether a row contains a value suitable for valueQuantity/valueString."""
        if "result" not in row:
            return False
        result_val = row.get("result")
        if result_val is None:
            return False
        value_str = str(result_val).strip()
        if not value_str:
            return False
        if value_str == "Выполнено. Отдельный бланк.":
            return False
        return True

    def convert_normalized_result_to_fhir(
        self, normalized_result: ReportNormalizedResult
    ) -> List[Bundle]:
        if not normalized_result.items:
            return []
        try:
            return self.convert_to_fhir({"tables": normalized_result.model_dump()["items"]})
        except Exception as e:
            logger.error(f"Failed to convert to fhir recognition results: {e}")
            return []
