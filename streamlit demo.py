import os

import streamlit as st
from openai import OpenAI

# Default instructions that shape the assistant's tone.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Keep responses concise and actionable."
)

# Attempt to read a value from Streamlit secrets without failing the app.
def read_secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return None

# Available model choices displayed in the sidebar.
MODEL_OPTIONS = {
    "GPT-4o mini (fast, cost-efficient)": "gpt-4o-mini",
    "GPT-4o (higher quality)": "gpt-4o",
}

st.set_page_config(page_title="ChatBot", page_icon=":speech_balloon:")
st.title("AI ChatBot")

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
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=120,
    )
    st.session_state.system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT

    if st.button("Clear conversation"):
        st.session_state.messages = []

# Initialize chat history storage.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Resolve API key priority: sidebar input > Streamlit secrets > environment.
api_key = (
    api_key_input
    or read_secret("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

if not api_key:
    st.info(
        "Add your OpenAI API key in the sidebar or set OPENAI_API_KEY before using the chat."
    )
    st.stop()

client = OpenAI(api_key=api_key)
model = MODEL_OPTIONS[model_label]

# Render conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box drives the interaction loop.
if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=model,
                stream=True,
                messages=[
                    {"role": "system", "content": st.session_state.system_prompt},
                    *st.session_state.messages,
                ],
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
            full_response = full_response.strip()
            if not full_response:
                full_response = "_No response received._"

            message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
