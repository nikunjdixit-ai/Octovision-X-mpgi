import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = os.getenv("APP_NAME", "OctoVizn X MPGI Backend")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "placeholder_secret")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"

settings = Settings()
