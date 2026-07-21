from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from .env / environment variables."""

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/paper_reader.db"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "paper-images"
    minio_secure: bool = False

    # arXiv
    arxiv_cache_dir: str = "./data/arxiv_cache"

    # LLM (OpenAI-compatible)
    openai_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.1

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-ci"
    gemini_base_url: str = "https://happyapi.org/v1"

    # Poster vision / review
    poster_vision_provider: str = "agnes"
    agnes_api_key: str = ""
    agnes_model: str = "agnes-2.0-flash"
    agnes_base_url: str = "https://apihub.agnes-ai.com/v1"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
