# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .orchestrator import Orchestrator

app = FastAPI(title="LLM State Manager")
orc = Orchestrator()


class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    active_model: str = "gemini-2.5-flash"
    k: int = 5
    last_n: int = 5


@app.post("/chat")
def chat(req: ChatRequest):
    return orc.chat(
        session_id=req.session_id,
        user_message=req.user_message,
        active_model=req.active_model,
        k=req.k,
        last_n=req.last_n,
    )


@app.post("/reset")
def reset(session_id: str):
    orc.reset_session(session_id)
    return {"ok": True}


@app.get("/models")
def models():
    return {"models": orc.available_models()}
