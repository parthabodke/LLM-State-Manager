# backend/providers.py
import os
from typing import List, Dict
import google.genai as genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Provider Initializations ---

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    _gemini_ok = True
else:
    _gemini_ok = False

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Groq
try:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    _groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    _groq_client = None

# --- Discovery Logic ---

def get_live_gemini_models() -> List[str]:
    if not _gemini_ok:
        return []
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        models = client.models.list()
        return [
            m.name.replace('models/', '')
            for m in models
            if 'generateContent' in (m.supported_actions or [])
        ]
    except Exception as e:
        print(f"⚠️ Error fetching Gemini models: {e}")
        return []

def get_live_groq_models() -> List[str]:
    """Fetches available models from Groq."""
    if _groq_client is None:
        return []
    try:
        models = _groq_client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        print(f"⚠️ Error fetching Groq models: {e}")
        return []

# --- Call Logic ---

def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

# ✅ New call_gemini
def call_gemini(model: str, messages: List[Dict[str, str]]) -> str:
    if not _gemini_ok:
        raise RuntimeError("Gemini API key not set.")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = _messages_to_text(messages)
        response = client.models.generate_content(
            model=model.replace('models/', ''),
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

def call_openai(model: str, messages: List[Dict[str, str]]) -> str:
    if _openai_client is None:
        raise RuntimeError("OpenAI API key not set.")
    resp = _openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def call_groq(model: str, messages: List[Dict[str, str]]) -> str:
    if _groq_client is None:
        raise RuntimeError("Groq not configured.")
    response = _groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def call_model(active_model: str, messages: List[Dict[str, str]]) -> str:
    name = active_model.lower().strip()
    try:
        # Routing logic
        if name.startswith("gemini"):
            return call_gemini(active_model, messages)
        elif name.startswith("gpt"):
            return call_openai(active_model, messages)
        # Check if the model is a Gemma model or any other hosted on Groq
        elif any(brand in name for brand in ["llama", "mixtral", "gemma", "deepseek", "qwen"]):
            return call_groq(active_model, messages)
        else:
            # Fallback
            try: return call_gemini(active_model, messages)
            except: return call_openai(active_model, messages)
    except Exception as e:
        return f"Model {active_model} is currently unavailable. Error: {str(e)}"