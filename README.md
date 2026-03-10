# LLM State Manager

> A conversational AI platform with persistent memory, multi-model support, and factual auditing — powered by ChromaDB, Google Gemini, Groq, and OpenAI.

**🚀 Live Demo → [llm-state-manager.streamlit.app](https://llm-state-manager.streamlit.app/)**

---

## What It Does

LLM State Manager gives your AI conversations a long-term memory. Instead of each message being stateless, the system:

- **Stores** every conversation turn as a vector embedding in ChromaDB
- **Retrieves** the most relevant past messages using semantic similarity (RAG)
- **Injects** that context into each new prompt automatically
- **Audits** numerical claims using a secondary Groq model for factual accuracy

---

## Architecture

```
User ──► Streamlit Frontend (ui.py)
              │
              ▼
         FastAPI Backend (main.py)
              │
         ┌────┴─────────────────┐
         │                      │
    Orchestrator           ChromaDB
    (orchestrator.py)    (vector memory)
         │
    ┌────┴────────────────────┐
    │                         │
  LLM Providers          Auditor
  (providers.py)         (Groq / gemma-3-4b-it)
  Gemini / GPT / Groq
```

---

## Features

| Feature | Description |
|---|---|
| 🔁 **Persistent Memory** | Past messages stored as embeddings, retrieved per session |
| 🤖 **Multi-Model Support** | Google Gemini, OpenAI GPT, Groq models |
| 🔍 **RAG Context** | Semantic search over chat history to inject relevant context |
| ✅ **Auditor** | Secondary model verifies numerical accuracy of responses |
| 🗂️ **Session Management** | Isolated memory per session ID |
| 🐳 **Docker Ready** | Full Docker + docker-compose setup included |

---

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized setup)
- API keys for at least one LLM provider:
  - [Google Gemini](https://aistudio.google.com) — `GEMINI_API_KEY`
  - [Groq](https://console.groq.com) — `GROQ_API_KEY` (also used by Auditor)
  - [OpenAI](https://platform.openai.com) — `OPENAI_API_KEY`

---

## Quickstart

### Run Locally

**1. Clone and set up environment**
```bash
git clone https://github.com/parthabodke/LLM-State-Manager.git
cd LLM-State-Manager
```

**2. Install backend dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**3. Install frontend dependencies**
```bash
cd ../frontend
pip install -r requirements.txt
```

**4. Create your `.env` in the project root**
```env
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
```

**5. Start the backend** (from `/backend` directory)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**6. Start the frontend** (from `/frontend` directory, in a new terminal)
```bash
BACKEND_URL=http://localhost:8000 streamlit run ui.py
```

**7. Open** http://localhost:8501

---

## Project Structure

```
LLM-State-Manager/
├── backend/
│   ├── main.py              # FastAPI app, API routes
│   ├── orchestrator.py      # RAG pipeline, context injection
│   ├── memory.py            # ChromaDB vector store logic
│   ├── providers.py         # Gemini, GPT, Groq API wrappers
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── ui.py                # Streamlit chat interface
│   ├── api_client.py        # HTTP client for backend
│   └── requirements.txt
├── docker-compose.yml
└── .env                     # ← Not committed, create manually
```

---

## Using the App

### 1. Start a Session
Enter a **Session ID** in the sidebar — this scopes your memory. Use any string (e.g. `my-session`, `project-alpha`).

### 2. Choose a Model
Select your LLM from the dropdown. Available models are fetched live from each provider.

### 3. Configure Memory
- **Top-K**: How many semantically similar past messages to retrieve
- **Last-N**: How many recent messages to always include

### 4. Enable the Auditor (Optional)
Toggle **"Enable Auditor"** in the sidebar. When active, a secondary Groq model (`gemma-3-4b-it`) checks each response for numerical accuracy and displays:
- ✅ Green badge — response passed factual check
- ⚠️ Red badge — potential inaccuracies detected

### 5. Reset Session
Click **"Reset Session"** to clear the memory for the current session ID.

---

## Deployment

The live demo is deployed using:

| Layer | Platform |
|---|---|
| **Backend** | [Render](https://render.com) — Docker Web Service |
| **Frontend** | [Streamlit Community Cloud](https://share.streamlit.io) |
| **Memory** | ChromaDB persistent volume on Render |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Recommended | Google Gemini API key |
| `GROQ_API_KEY` | Recommended | Groq API key (also powers Auditor) |
| `OPENAI_API_KEY` | Optional | OpenAI GPT API key |
| `BACKEND_URL` | Frontend only | URL of the FastAPI backend |

---

## Tech Stack

- **Backend**: FastAPI, ChromaDB, `google-genai`, `groq`, `openai`
- **Frontend**: Streamlit
- **Embeddings**: `gemini-embedding-001` via Google GenAI
- **Vector DB**: ChromaDB (cosine similarity, SQLite metadata)
- **Auditor**: `gemma-3-4b-it` via Groq
- **Containerization**: Docker, Docker Compose
