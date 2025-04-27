import os
from openai import AsyncOpenAI


def create_openai_client() -> AsyncOpenAI:
    """Return an instance of the OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=api_key)