# backend/memory.py
import os
import uuid
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)


class MemoryStore:
    """
    Conversation memory backed by ChromaDB.
    Uses local SentenceTransformer embeddings (free).
    """

    def __init__(self, collection_name: str = "chat_memory"):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Local, no API needed
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

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
        """
        Returns a list of past messages (role/content) most relevant to query,
        filtered by session_id.
        """
        if not query.strip():
            return []

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
        """
        Chroma is not an append-only log, so there's no strict ordering.
        This helper just returns up to last_n items via a broad get.
        """
        got = self.collection.get(where={"session_id": session_id})
        docs = []
        if got and got.get("ids"):
            for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
                docs.append({"role": meta.get("role", "unknown"), "content": doc})
        # Not truly chronological; best-effort recent window
        return docs[-last_n:] if last_n else docs

    def clear_session(self, session_id: str) -> None:
        got = self.collection.get(where={"session_id": session_id})
        if got and got.get("ids"):
            self.collection.delete(ids=got["ids"])
