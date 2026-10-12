import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from app.config.settings import EMBEDDING_MODEL, PATHS, NEWS_COLLECTION, DICT_COLLECTION

import numpy as np


_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def get_client():
    return chromadb.PersistentClient(path=str(PATHS["chroma"]))

def index_news(articles: list[dict]):
    client = get_client()
    col = client.get_or_create_collection(NEWS_COLLECTION)
    embedder = get_embedder()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    ids, docs, metas, embs = [], [], [], []
    for a in articles:
        chunks = splitter.split_text(a["title"] + "\n" + a["summary"])
        if not chunks:
            continue
        vectors = embedder.encode(chunks).tolist()
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            ids.append(f"{a['document_id']}_{i}")
            docs.append(chunk)
            metas.append({
                "source": a["url"],
                "title": a["title"],
                "published": a.get("published", ""),
                "type": "news",
            })
            embs.append(vec)

    if ids:
        col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    print(f"Indexed {len(ids)} news chunks.")

def index_dictionary_chunks(chunks: list, batch_size: int = 4000):
    """Index dictionary chunks in batches (ChromaDB has a 5461-item limit per add)."""
    client = get_client()
    col = client.get_or_create_collection(DICT_COLLECTION)
    embedder = get_embedder()

    texts = [c.page_content for c in chunks]
    if not texts:
        print("No dictionary chunks to index.")
        return

    # --- Embed (with disk cache) ---
    cache_path = PATHS["chroma"] / "dict_embeddings.npy"
    if cache_path.exists() and np.load(cache_path).shape[0] == len(texts):
        print(f"Loading cached embeddings from {cache_path}")
        vectors = np.load(cache_path).tolist()
    else:
        print(f"Embedding {len(texts)} chunks (this may take a while)...")
        vectors = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(cache_path, vectors)
        print(f"Cached embeddings to {cache_path}")
        vectors = vectors.tolist()

    ids = [f"dict_{i}" for i in range(len(texts))]
    metas = [{
        "dictionary_name": c.metadata.get("dictionary_name", ""),
        "page": c.metadata.get("page", 0),
        "type": "dictionary",
    } for c in chunks]

    # --- Insert in batches ---
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        col.add(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=texts[start:end],
            metadatas=metas[start:end],
        )
        print(f"  Inserted {end}/{total}")

    print(f"✓ Indexed {total} dictionary chunks.")

def load_processed_articles() -> list[dict]:
    path = PATHS["processed"] / "documents.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]
