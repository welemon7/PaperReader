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

    # LLM (Unified) - 统一使用这一个配置
    llm_api_key: str = "sk-X2AUiiAUKx1EV7qmFSr8NeErKFMpzAY1KxLeiV7ks5O5gYId"  # 统一API密钥
    llm_model: str = "agnes-2.0-flash"  # 统一模型
    llm_base_url: str = "https://apihub.agnes-ai.com/v1"  # 统一Base URL
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.1

    # 保留向后兼容（可选，可以删除）
    # openai_api_key: str = ""  # 可以删除
    # planner_api_key: str = ""  # 可以删除
    # gemini_api_key: str = ""  # 可以删除
    # agnes_api_key: str = ""  # 可以删除
    # poster_vision_provider: str = "agnes"  # 可以删除

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()