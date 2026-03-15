import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from endee import Endee

load_dotenv()

ENDEE_URL   = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME  = os.getenv("ENDEE_INDEX_NAME", "medical_reports")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 3

# ── Lazy singletons ────────────────────────────────────────────────────────

_model = None
_index = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_index():
    global _index
    if _index is None:
        client = Endee()
        client.set_base_url(f"{ENDEE_URL}/api/v1")
        if ENDEE_TOKEN:
            client.set_auth_token(ENDEE_TOKEN)
        _index = client.get_index(name=INDEX_NAME)
    return _index


# ── Core functions ─────────────────────────────────────────────────────────

def embed_query(question: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode([question], normalize_embeddings=True)[0]
    return vector.tolist()


def vector_search(query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
    index = get_index()
    results = index.query(vector=query_vector, top_k=top_k)

    # Debug: print raw type on first run so we can confirm field names
    if results:
        r0 = results[0]
        print(f"[Debug] result type={type(r0).__name__}  vars={vars(r0) if hasattr(r0, '__dict__') else r0}")

    chunks = []
    for r in results:
        # Support both dict-style and object-style results
        if isinstance(r, dict):
            meta       = r.get("meta", {}) or {}
            similarity = r.get("similarity", r.get("score", r.get("distance", 0.0)))
        else:
            meta       = getattr(r, "meta", None) or getattr(r, "metadata", None) or {}
            similarity = (getattr(r, "similarity", None)
                          or getattr(r, "score", None)
                          or getattr(r, "distance", 0.0))

        if isinstance(meta, dict):
            text        = meta.get("text", "")
            filename    = meta.get("filename", "unknown")
            chunk_index = meta.get("chunk_index", 0)
        else:
            text, filename, chunk_index = "", "unknown", 0

        chunks.append({
            "text":        text,
            "filename":    filename,
            "chunk_index": chunk_index,
            "similarity":  round(float(similarity), 4),
        })
    return chunks


# ── Public API ──────────────────────────────────────────────────────────────

def search(question: str, top_k: int = TOP_K) -> list[dict]:
    query_vector = embed_query(question)
    return vector_search(query_vector, top_k=top_k)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is the patient's diagnosis?"
    print(f"Question : {question}\n")
    results = search(question)
    for i, chunk in enumerate(results, 1):
        print(f"[{i}] {chunk['filename']}  (similarity={chunk['similarity']})")
        print(f"    {chunk['text'][:200]}…\n")
