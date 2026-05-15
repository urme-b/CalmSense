from pydantic_settings import BaseSettings


class ApiSettings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Models
    models_dir: str = "./models"
    default_model: str = ""

    # CORS
    cors_origins: str = ""

    # Auth
    jwt_secret_key: str = "calmsense-dev-secret"
    jwt_expire_seconds: int = 3600
    auth_enabled: bool = False
    api_key: str = "calmsense-dev-key"

    # Rate limiting
    rate_limit_per_minute: int = 120

    # WebSocket limits
    ws_max_clients: int = 100
    ws_max_buffer_size: int = 10000
    ws_max_batch_samples: int = 1000
    ws_window_size: int = 700

    # CSV limits
    csv_max_size: int = 10 * 1024 * 1024
    csv_max_rows: int = 10000

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"


settings = ApiSettings()
