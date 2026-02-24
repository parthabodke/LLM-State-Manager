# streamlit_app/api_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def send_chat(session_id: str, user_message: str, active_model: str, k: int = 6, last_n: int = 4):
    url = f"{BACKEND_URL}/chat"
    payload = {
        "session_id": session_id,
        "user_message": user_message,
        "active_model": active_model,
        "k": k,
        "last_n": last_n,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def reset_session(session_id: str):
    url = f"{BACKEND_URL}/reset"
    r = requests.post(url, params={"session_id": session_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def list_models():
    url = f"{BACKEND_URL}/models"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json().get("models", [])
