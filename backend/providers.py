# backend/providers.py
import os
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

# --- Gemini ---
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_ok = True
else:
    _gemini_ok = False

# --- OpenAI ---
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- Groq ---
try:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    _groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    _groq_client = None
    print("⚠️ Groq library not installed. Run: pip install groq")


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    """Convert message list to plain text format for Gemini."""
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


def call_gemini(model: str, messages: List[Dict[str, str]]) -> str:
    """
    Call Gemini API with proper model names.
    Important: DO NOT include 'models/' prefix in the model name!
    """
    if not _gemini_ok:
        raise RuntimeError("Gemini requested but GEMINI_API_KEY is not set.")
    
    # Remove 'models/' prefix if present
    if model.startswith('models/'):
        model = model.replace('models/', '')
    
    # List of known working Gemini models (as of 2025)
    # Based on Gemini 2.5 generation
    valid_models = [
        # --- Google Gemini API (Free Tier) ---
        # Frontier & Reasoning Tier
        'gemini-3.1-pro-preview',         # Current top-tier reasoning (Feb 2026)
        'gemini-3-pro-preview',           # Balanced multimodal flagship
        'gemini-3-flash-preview',         # Workhorse for speed and vision
        'gemini-3-deep-think',            # Science/math reasoning specialist
        'gemini-2.5-pro',                 # Stable production pro
        'gemini-2.5-flash',               # Stable production flash
        'gemini-2.5-flash-lite',          # Most generous free quota (1,000 RPD)
    
        # Gemma Series (Open Weights via Google AI Studio)
        'gemma-3-27b-it',                 # High-capability open model
        'gemma-3-12b-it',                 # Balanced efficiency
        'gemma-3-4b-it',                  # Multimodal-capable lightweight
        'gemma-3-1b-it',                  # Optimized for speed
        'gemma-3-270m-it',                # Ultra-compact, text-focused
        'gemma-3n-e4b-it',                # Mobile-first architecture (low-latency)
        'gemma-3n-e2b-it',                # Efficient mobile variant
    
        # --- Groq API (Developer Tier) ---
        # Performance & Low Latency
        'llama-4-scout-17b-instruct',     # Frontier Groq speed (750+ T/sec)
        'llama-3.3-70b-versatile',        # Best high-parameter open model on Groq
        'llama-3.1-8b-instant',           # Baseline for sub-100ms responses
        'deepseek-r1-distill-llama-70b',  # Chain-of-thought logic expert
        'qwen3-32b',                      # Coding and math specialist
        'openai/gpt-oss-120b',            # High-capacity open weight alternative
        'groq/compound',                  # Agentic model with built-in tool use
        'groq/compound-mini',             # Faster agentic variant
        
        # Audio Specialist
        'whisper-large-v3-turbo',
    ]
    
    # Validate model name
    if model not in valid_models:
        print(f"⚠️ Warning: '{model}' might not be available. Valid models: {valid_models}")
    
    try:
        # Convert messages to text prompt
        prompt = _messages_to_text(messages)
        
        # Create model instance WITHOUT 'models/' prefix
        model_obj = genai.GenerativeModel(model)
        
        # Generate content
        resp = model_obj.generate_content(prompt)
        
        # Extract text from response
        if hasattr(resp, 'text'):
            return resp.text.strip()
        elif hasattr(resp, 'parts') and resp.parts:
            return ''.join(part.text for part in resp.parts).strip()
        else:
            return str(resp).strip()
            
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful error message
        if "404" in error_msg or "not found" in error_msg:
            available = []
            try:
                # Try to list available models
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        clean_name = m.name.replace('models/', '')
                        available.append(clean_name)
                
                if available:
                    raise RuntimeError(
                        f"Model '{model}' not found. Available models: {', '.join(available[:5])}"
                    )
            except:
                pass
        
        raise RuntimeError(f"Gemini API error: {error_msg}")


def call_openai(model: str, messages: List[Dict[str, str]]) -> str:
    """Call OpenAI API."""
    if _openai_client is None:
        raise RuntimeError("OpenAI requested but OPENAI_API_KEY is not set.")
    
    resp = _openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def call_groq(model: str, messages: List[Dict[str, str]]) -> str:
    '''
    Call Groq API - extremely fast inference.
    Uses OpenAI-compatible format.
    '''
    if _groq_client is None:
        raise RuntimeError("Groq requested but GROQ_API_KEY is not set or groq library not installed.")
    
    response = _groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def call_model(active_model: str, messages: List[Dict[str, str]]) -> str:
    """
    Route to Gemini or OpenAI based on model name.
    Examples:
      - "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"  -> Gemini
      - "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo" -> OpenAI
    """
    name = active_model.lower().strip()

    try:
        if name.startswith("gemini"):
            return call_gemini(active_model, messages)
        elif name.startswith("gpt"):
            return call_openai(active_model, messages)
        elif name.startswith("llama") or name.startswith("mixtral") or name.startswith("gemma-2"):
            return call_groq(active_model, messages)
        else:
            # Default: try Gemini first (since it's free), then OpenAI
            try:
                return call_gemini(active_model, messages)
            except Exception:
                return call_openai(active_model, messages)
    except Exception as e:
        # Make error user-friendly
        return f"Provider error for `{active_model}`: {str(e)}"