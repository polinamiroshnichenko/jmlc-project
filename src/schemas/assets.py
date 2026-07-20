from pydantic import BaseModel, ConfigDict, Field

from src.schemas.chat import Coding


class Asset(BaseModel):
    coding: list[Coding] = Field(default_factory=list)
    text: str = Field(default="")


class NormalizerData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    gender: str | None = None
    code: str
    codesystem: str
    normalize_type: str = Field(alias="type")


class PartItem(BaseModel):
    code: str
    caption: str


class Complex(BaseModel):
    caption: str
    part_items: list[PartItem]


class CatalogMappingItem(BaseModel):
    hxid: str
    report_name: str


class NomenclatureComponent(BaseModel):
    component: str
    bio_name: str


class NomenclatureItem(BaseModel):
    components: list[NomenclatureComponent]
    commercial_name: str


class UCUMCode(BaseModel):

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(alias="ID")
    fullname: str = Field(alias="FULLNAME")
    shortname: str = Field(alias="SHORTNAME")
    ucum: str = Field(alias="UCUM")


class UCUMRecords(BaseModel):
    records: list[UCUMCode]
