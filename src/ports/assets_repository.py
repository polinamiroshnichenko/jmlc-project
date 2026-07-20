from typing import Protocol

from src.schemas.assets import (
    Asset,
    CatalogMappingItem,
    Complex,
    NomenclatureItem,
    NormalizerData,
    UCUMRecords,
)
from src.schemas.chat import InitiatorType


class AssetsRepository(Protocol):

    def biomaterials(self) -> dict[str, Asset]: ...

    def labnames(self) -> dict[str, Asset]: ...

    def laboratories(self) -> dict[str, Asset]: ...

    def ucum_codes(self) -> UCUMRecords: ...

    def nomenclatures(self) -> dict[str, NomenclatureItem]: ...

    def normalizer_data(self, initiator: InitiatorType | None = None) -> list[NormalizerData]: ...

    def complex_hxids(self) -> dict[str, Complex]: ...

    def specimen_container_catalog_mapping(
        self,
    ) -> dict[str, dict[str, list[CatalogMappingItem]]]: ...
