from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class UnmatchedPolicy(StrEnum):
    OTHER = "other"
    ERROR = "error"
    LEAVE = "leave"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FILE_ORGANIZER_",
        extra="ignore",
    )

    ollama_host: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_HOST")
    embed_model: str = "nomic-embed-text"
    vision_model: str = "moondream"
    naming_model: str | None = None
    existing_folder_threshold: float = 0.68
    existing_folder_margin: float = 0.06
    new_cluster_threshold: float = 0.72
    min_cluster_size: int = 2
    unmatched_policy: UnmatchedPolicy = UnmatchedPolicy.OTHER
    other_folder: str = "Other"
    text_head_chars: int = 4000


def load_settings(env_file: Path | None = None) -> Settings:
    if env_file is None:
        return Settings()
    return Settings(_env_file=env_file)
