# streamlit_app/ui.py
import streamlit as st
from .api_client import send_chat, reset_session, list_models


def run_ui():
    st.set_page_config(page_title="LLM State Manager", layout="wide")
    st.title("LLM State Manager")
    st.caption("Switch between Llama and Gemini while preserving context via ChromaDB.")

    # Sidebar
    st.sidebar.header("Settings")
    session_id = st.sidebar.text_input("Session ID", value="session-1")
    k = st.sidebar.slider("Semantic retrieval (k)", 1, 12, 6)
    last_n = st.sidebar.slider("Recent window (last_n)", 0, 10, 4)

    # Models
    try:
        models = list_models()
    except Exception as e:
        models = ["gemini-1.5-flash", "gemini-1.5-pro", "gpt-4o-mini", "gpt-3.5-turbo"]
        st.sidebar.warning(f"Could not fetch models from backend: {e}")

    active_model = st.sidebar.selectbox("Active model", models, index=0)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Reset Session"):
            try:
                reset_session(session_id)
                st.session_state["chat"] = []
                st.toast("Session memory cleared.")
            except Exception as e:
                st.error(f"Reset failed: {e}")
    with col2:
        show_ctx = st.checkbox("Show retrieved context", value=False)

    # Chat state
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    # Render history
    for role, content in st.session_state["chat"]:
        with st.chat_message(role):
            st.markdown(content)

    # Input
    user_message = st.chat_input("Type your message…")

    if user_message:
        st.session_state["chat"].append(("user", user_message))
        with st.chat_message("user"):
            st.markdown(user_message)

        try:
            resp = send_chat(session_id, user_message, active_model, k=k, last_n=last_n)
            answer = resp.get("answer", "")
            used_context = resp.get("used_context", [])
            model_used = resp.get("model_used", "unknown")

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"Model: `{model_used}`")
                if show_ctx and used_context:
                    with st.expander("Retrieved context"):
                        for item in used_context:
                            st.markdown(f"**{item.get('role','?')}:** {item.get('content','')}")

            st.session_state["chat"].append(("assistant", answer))

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Request failed: {e}")
