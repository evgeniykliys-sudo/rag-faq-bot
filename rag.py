import os
from pathlib import Path

import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "faq"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = os.getenv("RAG_MODEL") or "claude-haiku-4-5-20251001"
TOP_K = 3

SYSTEM_PROMPT = """Ты — ассистент поддержки внутренней AI-платформы.
Отвечай ТОЛЬКО на основе фрагментов документов, которые передаются в контексте.
Если в контексте нет ответа на вопрос — прямо скажи, что не нашёл информацию по этой теме
в базе знаний, и не придумывай ответ. Отвечай кратко и по делу, на русском языке."""

_embedder = None
_client = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_claude() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def search(question: str, top_k: int = TOP_K) -> list[dict]:
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": text, "source": meta["source"]})
    return chunks


def answer(question: str) -> tuple[str, list[str]]:
    chunks = search(question)

    if not chunks:
        return "База знаний пуста — сначала нужно запустить ingest.py", []

    context = "\n\n".join(f"[Источник: {c['source']}]\n{c['text']}" for c in chunks)
    sources = sorted(set(c["source"] for c in chunks))

    response = _get_claude().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Контекст из базы знаний:\n\n{context}\n\nВопрос: {question}",
            }
        ],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    return text, sources
