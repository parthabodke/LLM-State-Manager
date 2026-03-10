# frontend/ui.py
import streamlit as st
import sys
import os

# Add the current directory to sys.path to ensure absolute imports work
sys.path.append(os.path.dirname(__file__))

from api_client import send_chat, reset_session, list_models

def run_ui():
    st.set_page_config(page_title="LLM State Manager", layout="wide")
    st.title("LLM State Manager")
    st.caption("Universal Context Bridge: Switch models seamlessly without losing context.")

    # --- SIDEBAR ---
    st.sidebar.header("Configuration")
    session_id = st.sidebar.text_input("Session ID", value="session-1", help="Unique ID for your conversation")
    
    st.sidebar.subheader("Memory Settings")
    k = st.sidebar.slider("Semantic retrieval (k)", 1, 12, 6, help="How many relevant past messages to pull from ChromaDB")
    last_n = st.sidebar.slider("Recent window (last_n)", 0, 10, 4, help="How many immediate past messages to include for flow")

    st.sidebar.subheader("Model Selection")
    
    if "available_models" not in st.session_state:
        try:
            st.session_state["available_models"] = list_models()
        except Exception as e:
            st.session_state["available_models"] = ["gemini-2.5-flash-lite"]
            st.sidebar.error(f"Backend offline: {e}")

    if st.sidebar.button("Refresh Model List"):
        try:
            st.session_state["available_models"] = list_models()
            st.toast("Model list updated!")
        except Exception as e:
            st.error("Failed to sync with backend.")

    active_model = st.sidebar.selectbox(
        "Active model", 
        st.session_state["available_models"], 
        index=0,
        help="The 'brain' that will process your next message"
    )

    st.sidebar.divider()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Reset Session"):
            try:
                reset_session(session_id)
                st.session_state["chat"] = []
                st.toast("Memory wiped.")
            except Exception as e:
                st.error(f"Reset failed: {e}")
    with col2:
        show_ctx = st.checkbox("Show Context", value=False)
        
    st.sidebar.subheader("Advanced Features")
    use_auditor = st.sidebar.checkbox(
        "Enable Auditor", 
        value=True, 
        help="Runs a secondary model to verify numerical accuracy. Increases latency."
    )

    # --- CHAT INTERFACE ---
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    for chat_item in st.session_state["chat"]:
        role = chat_item["role"]
        content = chat_item["content"]
        v_data = chat_item.get("verification", {}) # Get the specific verification for THIS message
        
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                # Only show the badge if THIS specific message had auditing active
                if v_data.get("is_active") is True:
                    st.caption(f"Verified (Conf: {v_data.get('confidence_score')})")

    user_message = st.chat_input("Type your message…")

    if user_message:
        st.session_state["chat"].append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        try:
            with st.spinner(f"Routing to {active_model}..."):
                resp = send_chat(
                    session_id, user_message, active_model, 
                    k=k, last_n=last_n, use_auditor=use_auditor
                )
            
            answer = resp.get("answer", "")
            verification = resp.get("verification", {})
            model_used = resp.get("model_used", active_model)

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"Source: `{model_used}`")
                
                # --- THE STRICT CHECK ---
                # Using 'is True' ensures we don't trigger on empty dicts or 0.0 defaults
                if verification and verification.get("is_active") is True:
                    is_factual = verification.get("is_factual", True)
                    conf_score = verification.get("confidence_score", 0.0)
                    
                    if not is_factual:
                        st.error(f"**Numerical Discrepancy** (Confidence: {conf_score})")
                        with st.expander("Audit Logs"):
                            for issue in verification.get("errors", []):
                                st.write(f"- {issue}")
                    else:
                        st.success(f"**Verified against source data** (Confidence: {conf_score})")

            # CRITICAL: Save the verification result INTO the chat history 
            # so old messages don't "borrow" the current verification state.
            st.session_state["chat"].append({
                "role": "assistant", 
                "content": answer, 
                "model": model_used,
                "verification": verification # Save this here!
            })

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Backend Connection Error: {e}")
                
run_ui()