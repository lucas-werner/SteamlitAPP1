from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Keep responses concise and actionable."
)

MODEL_OPTIONS = {
    "GPT-4o mini (fast, cost-efficient)": "gpt-4o-mini",
    "GPT-4o (higher quality)": "gpt-4o",
    "GPT-4.1 mini (latest preview)": "gpt-4.1-mini",
    "GPT-4.1 (latest preview)": "gpt-4.1",
    "o3-mini (reasoning focus)": "o3-mini",
}

PAGE_TITLE = "ChatBot"
PAGE_ICON = ":speech_balloon:"
APP_NAME = "AI ChatBot"

EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "documents"
