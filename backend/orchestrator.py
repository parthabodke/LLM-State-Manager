# backend/orchestrator.py
from typing import Dict, List

from .memory import MemoryStore
from .providers import call_model


class Orchestrator:
    """
    - Saves every message to Chroma (MemoryStore)
    - Retrieves relevant context for each user query
    - Calls the chosen provider (Gemini/GPT)
    """

    def __init__(self):
        self.mem = MemoryStore()

    def available_models(self) -> List[str]:
        """
        Return list of available models.
        Based on your API access (Gemini 2.5 generation).
        """
        return [
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

    def chat(
        self,
        session_id: str,
        user_message: str,
        active_model: str,
        k: int = 6,
        last_n: int = 4,
    ) -> Dict:
        # 1) Save user message
        self.mem.add_message(session_id, "user", user_message)

        # 2) Retrieve context
        retrieved = self.mem.retrieve_context(session_id, user_message, top_k=k)
        recent = self.mem.get_recent(session_id, last_n=last_n)
        # merge: retrieved first (semantic), then recent (recency)
        context_msgs = retrieved + [m for m in recent if m not in retrieved]

        # 3) Build messages
        messages: List[Dict[str, str]] = []
        # optional system prompt
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Use the conversation context if relevant, but avoid repetition."
        })
        messages.extend(context_msgs)
        messages.append({"role": "user", "content": user_message})

        # 4) Call model
        answer = call_model(active_model, messages)

        # 5) Save assistant answer
        self.mem.add_message(session_id, "assistant", answer)

        return {
            "answer": answer,
            "model_used": active_model,
            "used_context": context_msgs,
        }

    def reset_session(self, session_id: str) -> None:
        self.mem.clear_session(session_id)