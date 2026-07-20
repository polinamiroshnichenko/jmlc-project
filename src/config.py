import yaml
import json
import os
from typing import Any
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from consul import Consul


class ConsulSource(PydanticBaseSettingsSource):
    def __init__(
        self, settings_cls: type[BaseSettings], consul_client: Consul, prefix: str = ""
    ):
        super().__init__(settings_cls)
        self.client = consul_client
        self.prefix = (prefix or "").rstrip("/")

    def __call__(self):
        _, data = self.client.kv.get(self.prefix, recurse=True)

        if not data:
            return {}
        result: dict[str, Any] = {}

        for item in data:
            key: str = item["Key"]
            if self.prefix:
                key = key.removeprefix(self.prefix).lstrip("/")

            value = item["Value"]

            if value is None:
                continue
            value = value.decode()

            try:
                value = json.loads(value)
            except Exception:
                pass

            self.__insert_nested(result, key.split("/"), value)

        return result

    def get_field_value(self, field, field_name):
        raise NotImplementedError("ConsulSource not implement get field value")

    @staticmethod
    def __insert_nested(target: dict[str, Any], path: list[str], value: Any):
        current = target
        for part in path[:-1]:
            current = current.setdefault(part, {})

        current[path[-1]] = value


class YamlDirectoryConfigSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], configs_dir: str | Path):
        super().__init__(settings_cls)
        self.configs_dir = configs_dir

    def __call__(self):
        result = {}
        for file in sorted(self.configs_dir.glob("*.yaml")):
            with file.open() as f:
                data = yaml.safe_load(f) or {}

            result = self._deep_merge(result, data)
        return result

    @staticmethod
    def _deep_merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        result = base.copy()

        for k, v in new.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = YamlDirectoryConfigSettingsSource._deep_merge(result[k], v)
            else:
                result[k] = v

        return result

    def get_field_value(self, field, field_name):
        raise NotImplementedError(
            "get_field_value method is not implemented for YamlDirectoryConfigSettingsSource"
        )


class LLMConfig(BaseModel):
    base_url: str = Field(
        default_factory=lambda: ""
    )
    token: str


class TemporalWorkerConfig(BaseModel):
    max_concurrent_activities: int = Field(default_factory=lambda: 10)
    max_concurrent_workflow_tasks: int = Field(default_factory=lambda: 5)


class LabtestRecognitionConfig(BaseModel):
    url: str


class TerminologySettings(BaseModel):
    url: str = Field(default="https://terminology-v2-stage.example.com/api/v1")


class TemporalConfig(BaseModel):
    task_queue: str
    namespace: str
    address: str


class ServerSettings(BaseModel):
    addr: str = Field(default_factory=lambda: "0.0.0.0")
    port: int = Field(default_factory=lambda: 8000, gt=1, le=65535)
    root_path: str = Field(default="")


class MetricsSettings(BaseModel):
    enabled: bool = Field(default_factory=lambda: True)
    should_group_status_codes: bool
    should_ignore_untemplated: bool
    should_respect_env_var: bool
    should_instrument_requests_inprogress: bool
    excluded_handlers: list[str]
    inprogress_name: str
    inprogress_labels: bool


class McpSettings(BaseModel):
    enabled: bool = False
    server_url: str | None = None
    tool_name: str | None = None


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )
    root_dir: Path = Field(Path("/app"))
    temporal: TemporalConfig
    llm: LLMConfig
    labtest_recognition: LabtestRecognitionConfig
    worker: TemporalWorkerConfig
    logging: dict[str, Any] = Field(default_factory=lambda: {})
    server: ServerSettings
    metrics: MetricsSettings
    mcp: McpSettings = McpSettings()
    terminology: TerminologySettings = TerminologySettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):

        sources = [
            init_settings,
            YamlDirectoryConfigSettingsSource(
                settings_cls, Path(os.getenv("CONFIGS_DIR", "/etc/agents-workflow"))
            ),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ]
        consul_host = (os.getenv("CONSUL_HOST") or "").strip()
        if consul_host:
            client = Consul(
                host=consul_host,
                port=int(os.getenv("CONSUL_PORT") or 8500),
                token=os.getenv("CONSUL_TOKEN"),
                scheme=os.getenv("CONSUL_SCHEME") or "http",
                consistency=os.getenv("CONSUL_CONSISTENCY") or "default",
                dc=os.getenv("CONSUL_DC"),
                verify=(os.getenv("CONSUL_VERIFY") or "false").lower()
                in ("1", "true", "yes"),
                cert=os.getenv("CONSUL_CERT"),
            )
            sources.insert(
                2,
                ConsulSource(
                    settings_cls, client, prefix=os.getenv("CONSUL_PREFIX") or ""
                ),
            )
        return tuple(sources)


config = Config()
