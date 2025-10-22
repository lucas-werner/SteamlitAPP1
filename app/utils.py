import os
from typing import Optional

import streamlit as st


def read_secret(key: str) -> Optional[str]:
    """Safely fetch a value from Streamlit secrets."""
    try:
        return st.secrets[key]
    except Exception:
        return None


def resolve_api_key(input_key: Optional[str]) -> Optional[str]:
    """Pick the first available API key from input, secrets, or env vars."""
    return input_key or read_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def init_session_state(default_prompt: str) -> None:
    """Ensure session state carries prompt and conversation defaults."""
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = default_prompt
    if "messages" not in st.session_state:
        st.session_state.messages = []

