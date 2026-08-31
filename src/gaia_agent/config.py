from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):    

    app_env: str = "development"
    app_name: str = "gaia-agent"
    debug: bool = True
    database_url: str

    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    agent_max_iterations: int = 20
    context_max_tokens: int = 12000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()