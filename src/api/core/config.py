from pydantic_settings import BaseSettings,SettingsConfigDict

class Config(BaseSettings):

    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str | None = None
    CO_API_KEY: str | None = None
    MEETUP_ADMIN_PASSWORD: str = "humansftw"

    # Same source as meetup_context persistence (.env is loaded here; os.environ alone is not enough).
    # None → default /tmp/meetup_agent_state.json; "" → disable disk persistence.
    MEETUP_STATE_FILE: str | None = None

    model_config = SettingsConfigDict(env_file=".env")

config=Config()