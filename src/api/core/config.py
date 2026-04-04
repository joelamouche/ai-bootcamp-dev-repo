from pydantic_settings import BaseSettings,SettingsConfigDict

class Config(BaseSettings):

    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str | None = None
    CO_API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=".env")

config=Config()