import json
from pathlib import Path

from src.schemas.assets import (
    Asset,
    CatalogMappingItem,
    Complex,
    NomenclatureItem,
    NormalizerData,
    UCUMRecords,
)
from src.schemas.chat import InitiatorType


class FSAssetsRepository:
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir

        self._biomaterials: dict[str, Asset] = {
            k: Asset.model_validate(obj)
            for (k, obj) in self.__load_json("biomaterials.json").items()
        }

        self._labnames: dict[str, Asset] = {
            k: Asset.model_validate(obj) for (k, obj) in self.__load_json("labnames.json").items()
        }
        self._ucum_codes: UCUMRecords = UCUMRecords.model_validate(
            self.__load_json("UCUM_codes_nsi.json")
        )
        self._nomenclature: dict[str, NomenclatureItem] = {
            key: NomenclatureItem.model_validate(val)
            for (key, val) in self.__load_json("nomenclature.json").items()
        }
        self._laboratories: dict[str, Asset] = {
            k: Asset.model_validate(obj)
            for (k, obj) in self.__load_json("laboratories.json").items()
        }
        self._normalizer_data: dict[str, list[NormalizerData]] = {
            "default": [
                NormalizerData.model_validate(obj)
                for obj in self.__load_json("normalizer_data.json")
            ],
            "budzdorov": [
                NormalizerData.model_validate(obj)
                for obj in self.__load_json("budzdorov_normalizer_data.json")
            ],
        }

        self._complex_hxids: dict[str, Complex] = {
            key: Complex.model_validate(val)
            for (key, val) in self.__load_json("complexes.json").items()
        }
        self._catalog_mapping: dict[str, dict[str, list[CatalogMappingItem]]] = {
            ik: {
                jk: [CatalogMappingItem.model_validate(obj) for obj in jv]
                for (jk, jv) in iv.items()
            }
            for (ik, iv) in self.__load_json("specimen_container_catalog_mapping.json").items()
        }

    def __load_json(self, filename: str):
        with open(self.assets_dir / filename, "rb") as f:
            return json.loads(f.read())

    def biomaterials(self) -> dict[str, Asset]:
        return self._biomaterials

    def labnames(self) -> dict[str, Asset]:
        return self._labnames

    def laboratories(self) -> dict[str, Asset]:
        return self._laboratories

    def ucum_codes(self) -> UCUMRecords:
        return self._ucum_codes

    def nomenclatures(self) -> dict[str, NomenclatureItem]:
        return self._nomenclature

    def normalizer_data(self, initiator: InitiatorType | None = None) -> list[NormalizerData]:
        key = "budzdorov" if initiator == "budzdorov" else "default"
        return self._normalizer_data[key]

    def complex_hxids(self) -> dict[str, Complex]:
        return self._complex_hxids

    def specimen_container_catalog_mapping(self) -> dict[str, dict[str, list[CatalogMappingItem]]]:
        return self._catalog_mapping
