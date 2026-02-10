import os
from dotenv import load_dotenv

load_dotenv()  # loads .env once at import time

def get_env_var(key: str, default=None, required: bool = False):
    """
    Fetch an environment variable from .env or system env.

    Args:
        key (str): Environment variable name
        default: Default value if not found
        required (bool): Raise error if variable is missing

    Returns:
        str | None
    """
    value = os.getenv(key, default)

    if required and value is None:
        raise RuntimeError(f"Missing required environment variable: {key}")

    return value

