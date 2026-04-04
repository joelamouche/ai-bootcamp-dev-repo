from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str

    # Docker Compose: http://api:8000 (service `api`, container port 8000). Host port mapping does not apply here.
    API_URL: str = Field(default="http://api:8000", description="Base URL for the FastAPI service")

    model_config = SettingsConfigDict(env_file=".env")

config = Config()