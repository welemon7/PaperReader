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
    llm_api_key: str = ""  # 统一API密钥
    llm_model: str = ""  # 统一模型
    llm_base_url: str = ""  # 统一Base URL
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.1

    # Harness (视觉审查循环)
    harness_threshold: int = 9  # 达标分数 (0-10)
    harness_max_rounds: int = 5  # 最大审查轮数
    harness_enable_qa: bool = True  # 循环结束后运行 PaperQuiz 式内容评测
    harness_qa_threshold: float = 0.8  # 图像问答正确率门槛
    harness_zoom_crops: bool = True  # 是否给 VLM 发送 section 局部裁剪图
    harness_max_crops: int = 3  # 最多裁剪几个文本最密的 section
    harness_vision_model: str = ""  # 可选：独立视觉模型名；留空则复用 llm_model
    # 预留扩展点（本期不实现插图生成/插入）
    figure_gen_enabled: bool = False

    # 保留向后兼容（可选，可以删除）
    # openai_api_key: str = ""  # 可以删除
    # planner_api_key: str = ""  # 可以删除
    # gemini_api_key: str = ""  # 可以删除
    # agnes_api_key: str = ""  # 可以删除
    # poster_vision_provider: str = "agnes"  # 可以删除

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
