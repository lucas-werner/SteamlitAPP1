from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import streamlit as st


@dataclass
class SidebarConfig:
    api_key_input: str
    model_label: str
    system_prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    frequency_penalty: float
    presence_penalty: float


def render_sidebar(model_options: Sequence[str], default_prompt: str) -> SidebarConfig:
    with st.sidebar:
        st.header("Settings")
        api_key_input = st.text_input(
            "OpenAI API key",
            type="password",
            help=(
                "Leave blank if OPENAI_API_KEY is already set as an environment "
                "variable or in Streamlit secrets."
            ),
        )

        model_label = st.selectbox("Model", model_options)

        system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            height=120,
        )
        st.session_state.system_prompt = system_prompt.strip() or default_prompt

        if st.button("Clear conversation"):
            st.session_state.messages = []

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.05,
            help=(
                "Higher values increase creativity; lower values make responses more "
                "deterministic."
            ),
        )
        top_p = st.slider(
            "Top-p (nucleus sampling)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help=(
                "Sample from the smallest tokens set whose cumulative probability "
                "exceeds this value."
            ),
        )
        max_tokens = st.number_input(
            "Max tokens (0 for model default)",
            min_value=0,
            max_value=4096,
            value=0,
            step=32,
            help=(
                "Upper bound on tokens in the reply. Leave at 0 to let the model "
                "decide."
            ),
        )
        frequency_penalty = st.slider(
            "Frequency penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Higher values reduce repetition of lines in the response.",
        )
        presence_penalty = st.slider(
            "Presence penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Higher values increase the likelihood of new topics appearing.",
        )

    return SidebarConfig(
        api_key_input=api_key_input,
        model_label=model_label,
        system_prompt=st.session_state.system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=int(max_tokens),
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )


def render_document_status(
    errors: Iterable[str],
    vector_ready: bool,
    chunk_count: int,
    documents_dir_name: str,
) -> None:
    if errors:
        with st.sidebar.expander("Document warnings", expanded=False):
            for warning in errors:
                st.write(f"- {warning}")

    if vector_ready:
        st.sidebar.success(f"Knowledge base ready · {chunk_count} chunks indexed.")
    else:
        st.sidebar.info(
            f"Add .txt, .md, or .pdf files to {documents_dir_name}/ to enable "
            "knowledge-base answers."
        )


def render_context_expander(docs: Iterable) -> None:
    docs = list(docs)
    if not docs:
        return

    with st.expander("Knowledge base context", expanded=False):
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Document")
            source_name = Path(source).name if source else f"Document {idx}"
            st.markdown(f"**{idx}. {source_name}**")
            st.markdown(doc.page_content)

