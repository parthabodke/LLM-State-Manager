# backend/orchestrator.py
import re
import json
from typing import Dict, List
from memory import MemoryStore
from providers import call_model, get_live_gemini_models, get_live_groq_models

CRITIC_SYSTEM_PROMPT = """
You are a Financial Data Auditor. Your job is to verify the 'Proposed Answer' 
against the 'Retrieved Context'. 
1. Extract every number, date, and percentage from the Proposed Answer.
2. Check if those exact values exist in the Retrieved Context.
3. If a number is missing or different, flag it as a HALLUCINATION.
4. Return a JSON object: {"is_factual": boolean, "errors": [list of strings], "confidence_score": 0-1}
"""

class Orchestrator:
    def __init__(self):
        self.mem = MemoryStore()
        self._cached_models = []
        self.refresh_models()

    DEFAULT_MODEL = "gemini-2.5-flash"
    
    def refresh_models(self) -> List[str]:
        google_list = get_live_gemini_models()
        groq_list = get_live_groq_models()
        combined = list(set(google_list + groq_list))
        self._cached_models = sorted(combined)
        
        # Pre-select the default model after fetching
        self.active_model = next(
            (m for m in self._cached_models if "2.5-flash" in m),
            self._cached_models[0] if self._cached_models else self.DEFAULT_MODEL
        )
        return self._cached_models

    def available_models(self) -> List[str]:
        if not self._cached_models:
            return self.refresh_models()
        return self._cached_models

    def _parse_json(self, text: str) -> Dict:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                res = json.loads(match.group(0))
                return res if isinstance(res, dict) else {}
            return {"is_factual": True, "confidence_score": 0.5}
        except:
            return {"is_factual": True, "confidence_score": 0.0}

    def chat(
        self,
        session_id: str,
        user_message: str,
        active_model: str,
        k: int = 6,
        last_n: int = 4,
        use_auditor: bool = True,
    ) -> Dict:
        
        # 1. Save user message
        self.mem.add_message(session_id, "user", user_message)

        # 2. Retrieve context
        retrieved = self.mem.retrieve_context(session_id, user_message, top_k=k)
        recent = self.mem.get_recent(session_id, last_n=last_n)
        context_msgs = retrieved + [m for m in recent if m not in retrieved]

        # 3. Build messages
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": f"You are currently the {active_model} model."},
            {"role": "system", "content": "Use context if relevant, avoid repetition."}
        ]
        messages.extend(context_msgs)
        messages.append({"role": "user", "content": user_message})

        try:
            # 4. Generate Answer
            answer = call_model(active_model, messages)
            self.mem.add_message(session_id, "assistant", answer)

            # 5. Auditor Logic
            verification = {"is_active": False}
            if use_auditor:
                try:
                    critic_model = "gemma-3-4b-it" 
                    audit_input = f"CONTEXT: {context_msgs}\n\nPROPOSED ANSWER: {answer}"
                    audit_raw = call_model(critic_model, [
                        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                        {"role": "user", "content": audit_input}
                    ])
                    verification = self._parse_json(audit_raw)
                    verification["is_active"] = True 
                except Exception:
                    # If critic fails, we still mark active but default to safe/low confidence
                    verification = {"is_factual": True, "is_active": True, "confidence_score": 0.0}
            else:
                # Force inactive state so UI hides the box
                verification = {"is_active": False}

        except Exception as e:
            answer = f"Orchestrator Error: {str(e)}"
            #   verification = {"is_active": False}

        return {
            "answer": answer,
            "verification": verification,
            "model_used": active_model,
            "used_context": context_msgs,
        }

    def reset_session(self, session_id: str) -> None:
        self.mem.clear_session(session_id)