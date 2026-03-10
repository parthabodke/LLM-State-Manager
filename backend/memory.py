# backend/memory.py
import sys

# This fix is ONLY needed for Linux deployment (Render/GCP)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("DEBUG: Using pysqlite3 for ChromaDB compatibility.")
except ImportError:
    print("DEBUG: pysqlite3 not found, using standard sqlite3 (Expected on Windows).")

import os
import uuid
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions


# Define paths correctly
#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
#CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
# Force an absolute path that Docker understands
CHROMA_DIR = "/app/data/chroma"
if not os.path.exists(CHROMA_DIR):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
class MemoryStore:
    """
    Conversation memory backed by ChromaDB.
    Uses Google Gemini's text-embedding-004 model.
    """

    def __init__(self, collection_name: str = "gemini_state_manager"):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)

        # FIX: Changed model_name to "models/gemini-embedding-001"
        # In the 2026 API version, this is the most stable identifier for retrieval tasks.
        self.embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name="models/gemini-embedding-001" 
        )

        # FIX: Changed collection_name to "gemini_state_manager"
        # This prevents the 384 vs 768 dimension mismatch error from your old collection.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        _id = f"{session_id}_{role}_{uuid.uuid4().hex}"
        self.collection.add(
            ids=[_id],
            documents=[content],
            metadatas=[{"session_id": session_id, "role": role}],
        )

    def retrieve_context(
        self, session_id: str, query: str, top_k: int = 6
    ) -> List[Dict[str, str]]:
        if not query.strip():
            return []

        # Chroma calls the Gemini API here to embed your query automatically
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"session_id": session_id},
        )

        items: List[Dict[str, str]] = []
        if results and results.get("documents") and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                items.append({"role": meta.get("role", "unknown"), "content": doc})
        return items

    def get_recent(
        self, session_id: str, last_n: int = 6
    ) -> List[Dict[str, str]]:
        got = self.collection.get(where={"session_id": session_id})
        docs = []
        if got and got.get("ids"):
            for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
                docs.append({"role": meta.get("role", "unknown"), "content": doc})
        return docs[-last_n:] if last_n else docs

    def clear_session(self, session_id: str) -> None:
        got = self.collection.get(where={"session_id": session_id})
        if got and got.get("ids"):
            self.collection.delete(ids=got["ids"])