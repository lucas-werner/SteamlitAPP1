import streamlit as st
from openai import OpenAI

from app.config import (
    APP_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_SYSTEM_PROMPT,
    DOCS_DIR,
    EMBEDDING_MODEL,
    MODEL_OPTIONS,
    PAGE_ICON,
    PAGE_TITLE,
    RETRIEVAL_K,
)
from app.rag import build_vector_store, retrieve_documents
from app.ui import render_context_expander, render_document_status, render_sidebar
from app.utils import init_session_state, resolve_api_key

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.title(APP_NAME)

init_session_state(DEFAULT_SYSTEM_PROMPT)

sidebar_config = render_sidebar(list(MODEL_OPTIONS.keys()), DEFAULT_SYSTEM_PROMPT)

api_key = resolve_api_key(sidebar_config.api_key_input)
if not api_key:
    st.info(
        "Add your OpenAI API key in the sidebar or set OPENAI_API_KEY before using the chat."
    )
    st.stop()

client = OpenAI(api_key=api_key)
model = MODEL_OPTIONS[sidebar_config.model_label]

vector_store, doc_errors, chunk_count = build_vector_store(
    api_key=api_key,
    doc_dir=str(DOCS_DIR),
    embedding_model=EMBEDDING_MODEL,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

render_document_status(
    errors=doc_errors,
    vector_ready=vector_store is not None,
    chunk_count=chunk_count,
    documents_dir_name=DOCS_DIR.name,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    context_text, retrieved_docs, retrieval_error = retrieve_documents(
        vector_store=vector_store,
        prompt=prompt,
        k=RETRIEVAL_K,
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if retrieval_error:
            st.warning(f"Retrieval error: {retrieval_error}")
        else:
            render_context_expander(retrieved_docs)

        message_placeholder = st.empty()
        full_response = ""

        completion_settings = {
            "temperature": float(sidebar_config.temperature),
            "top_p": float(sidebar_config.top_p),
            "frequency_penalty": float(sidebar_config.frequency_penalty),
            "presence_penalty": float(sidebar_config.presence_penalty),
        }
        if sidebar_config.max_tokens > 0:
            completion_settings["max_tokens"] = int(sidebar_config.max_tokens)

        conversation_messages = [
            {"role": "system", "content": st.session_state.system_prompt},
            *st.session_state.messages,
        ]
        if context_text:
            conversation_messages.insert(
                1,
                {
                    "role": "system",
                    "content": (
                        "Use the following context to answer the user's latest "
                        f"question:\n\n{context_text}"
                    ),
                },
            )

        try:
            stream = client.chat.completions.create(
                model=model,
                stream=True,
                messages=conversation_messages,
                **completion_settings,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if not content:
                    continue

                full_response += content
                message_placeholder.markdown(full_response)
        except Exception as exc:
            st.error(f"OpenAI API error: {exc}")
        else:
            final_response = full_response.strip() or "_No response received._"
            message_placeholder.markdown(final_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )

