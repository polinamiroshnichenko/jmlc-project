from pydantic_settings import BaseSettings, YamlConfigSettingsSource


class AssistantConfig(BaseSettings):
    base_prompt: str = ""
    template: str = ""
    preinstruction_text: str = ""
    postinstruction_text: str = ""
    system_main: str = ""
    system_glossary: str = ""
    user_prompt: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "AssistantConfig":
        return cls(
            **YamlConfigSettingsSource(cls, yaml_file=path, yaml_file_encoding="utf-8").yaml_data
        )
