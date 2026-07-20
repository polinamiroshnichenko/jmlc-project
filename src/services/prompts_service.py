from pathlib import Path
from src.schemas.assistant import AssistantConfig
from src.config import config


class PromptsService:
    def __init__(self, prompts_dir: Path):

        self.prompts = dict([self.__read_config(file) for file in prompts_dir.rglob("**/*.yml")])

    @staticmethod
    def __read_config(path: Path) -> tuple[str, AssistantConfig]:
        return (path.name.rsplit(".")[0], AssistantConfig.from_yaml(str(path)))


prompts_service = PromptsService(config.root_dir / "prompts")
