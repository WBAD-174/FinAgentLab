# backend/app/core/config.py

from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    openai_api_key: str

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"

    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    chroma_dir: Path = BASE_DIR / "data" / "processed" / "chroma"

    collection_name: str = "finagentlab_docs"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()