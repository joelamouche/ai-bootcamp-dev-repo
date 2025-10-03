from pydantic_settings import BaseSettings,SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str
    GROQ_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

config=Config()