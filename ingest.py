from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "faq"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str) -> list[str]:
    """Режем по параграфам (двойной перенос строки) — годится для коротких FAQ-документов.

    Заголовок документа (первая строка, начинающаяся с #) не индексируется как
    отдельный чанк — сам по себе он почти не несёт смысла и в поиске вытесняет
    релевантные фрагменты. Вместо этого он добавляется как контекст к каждому
    содержательному чанку.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    if paragraphs[0].startswith("#"):
        title, body_paragraphs = paragraphs[0], paragraphs[1:]
        return [f"{title}\n{p}" for p in body_paragraphs]

    return paragraphs


def build_index():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(COLLECTION_NAME)

    ids, texts, metadatas = [], [], []

    for doc_path in sorted(DOCUMENTS_DIR.glob("*.md")):
        content = doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc_path.stem}_{i}")
            texts.append(chunk)
            metadatas.append({"source": doc_path.name})

    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    print(f"Проиндексировано {len(texts)} чанков из {len(list(DOCUMENTS_DIR.glob('*.md')))} документов")


if __name__ == "__main__":
    build_index()
