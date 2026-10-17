from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configuration in one typed place. Override any value via .env or environment variables."""

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # LLM (Groq exposes an OpenAI-compatible API)
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: float = 30.0

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking
    chunk_words: int = 200      # ~260 tokens, safely under the 512-token embedding limit
    chunk_overlap: int = 40

    # Retrieval
    default_top_k: int = 3
    max_top_k: int = 5
    alpha: float = 0.5          # weight of BM25 vs embeddings in the hybrid score
    min_similarity: float = 0.55  # tuned with evaluation/run_eval.py

    # Paths
    docs_dir: Path = BASE_DIR / "documents"
    cache_dir: Path = BASE_DIR / "cache"
    static_dir: Path = BASE_DIR / "static"


settings = Settings()