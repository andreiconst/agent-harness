import os

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-4-5"


def get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return Anthropic(api_key=api_key)
