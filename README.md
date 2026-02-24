# LLM-State-Manager

A simple and extensible **State Management System for LLM-powered applications** — designed to help developers persist, update, and manage conversational or application state across interactions with Large Language Models.

This repository includes a **backend API server** and a **Streamlit frontend** for interacting with stateful LLM workflows.

---

## 📌 Features

- 🧠 **Persistent LLM State:** Store and update conversation or application state to make LLM interactions more contextual.
- 🛠️ **Backend API:** A backend service to handle state operations (CRUD + sessions).
- 🌐 **Streamlit UI:** Lightweight UI to visualize and manipulate stateful data in real time.
- ⚡ **Modular Structure:** Easy to extend for your own LLM workflows or agent frameworks.

---

Clone the repository
```bash git clone https://github.com/parthabodke/LLM-State-Manager.git`
cd LLM-State-Manager```

Installation
Install dependencies:
`pip install -r requirements.txt`

Running the Backend
```bash cd backend`
uvicorn main:app --reload```
This launches the API server (default: http://127.0.0.1:8000).


Running the Frontend
From the project root:
`streamlit run streamlit_app/app.py`
This opens the Streamlit interface in your browser.
