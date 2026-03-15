import os
import sys

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from endee import Endee

load_dotenv()

ENDEE_URL    = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_TOKEN  = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME   = os.getenv("ENDEE_INDEX_NAME", "medical_reports")
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 3
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PROMPT_TEMPLATE = """\
You are a medical information assistant.
Use the provided medical report context to answer the question clearly.

Context:
{retrieved_chunks}

Question:
{user_question}

Answer in simple language."""

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


# ── Embedding ──────────────────────────────────────────────────────────────

def embed_query(question: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode([question], normalize_embeddings=True)[0]
    return vector.tolist()


# ── Vector search ──────────────────────────────────────────────────────────

def vector_search(query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
    index = get_index()
    results = index.query(vector=query_vector, top_k=top_k)

    chunks = []
    for r in results:
        if isinstance(r, dict):
            meta       = r.get("meta", {}) or {}
            similarity = r.get("similarity", 0.0)
        else:
            meta       = getattr(r, "meta", None) or getattr(r, "metadata", None) or {}
            similarity = getattr(r, "similarity", getattr(r, "score", 0.0))

        chunks.append({
            "text":        meta.get("text", ""),
            "filename":    meta.get("filename", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "similarity":  round(float(similarity), 4),
        })
    return chunks


# ── Context builder ────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['filename']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


# ── LLM generation ─────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "\n❌ GEMINI_API_KEY is not set.\n"
            "   Add it to your .env file:\n"
            "   GEMINI_API_KEY=your_key_here\n"
            "   Get a free key at: https://aistudio.google.com/"
        )
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


def generate_answer(context: str, question: str) -> str:
    prompt = PROMPT_TEMPLATE.format(
        retrieved_chunks=context,
        user_question=question,
    )
    return _call_gemini(prompt)


# ── Full RAG pipeline ──────────────────────────────────────────────────────

def rag(question: str, top_k: int = TOP_K) -> dict:
    # 1. Retrieve relevant chunks
    query_vector = embed_query(question)
    chunks       = vector_search(query_vector, top_k=top_k)

    if not chunks:
        return {
            "answer":  "No relevant documents found. Please ingest reports first.",
            "chunks":  [],
            "context": "",
        }

    # 2. Build context block
    context = build_context(chunks)

    # 3. Generate answer via LLM
    answer = generate_answer(context, question)

    return {
        "answer":  answer,
        "chunks":  chunks,
        "context": context,
    }


# Alias for simpler imports
def search(question: str, top_k: int = TOP_K) -> list[dict]:
    query_vector = embed_query(question)
    return vector_search(query_vector, top_k=top_k)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What does low hemoglobin mean?"
    print(f"Question : {question}")
    print(f"Provider : Gemini ({GEMINI_MODEL})\n{'─' * 50}")

    result = rag(question)

    print(f"\n🤖 Answer:\n{result['answer']}")
    print(f"\n📚 Sources ({len(result['chunks'])} chunks):")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"  [{i}] {chunk['filename']}  (similarity={chunk['similarity']})")
        print(f"       {chunk['text'][:120]}…")
