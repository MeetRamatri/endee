

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# ── Config ─────────────────────────────────────────────────────────────────
load_dotenv()

DATA_DIR        = Path(__file__).parent.parent / "data"
ENDEE_URL       = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_TOKEN     = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME      = os.getenv("ENDEE_INDEX_NAME", "medical_reports")
EMBED_MODEL     = "all-MiniLM-L6-v2"
VECTOR_DIM      = 384
WORDS_PER_CHUNK = 400


# ── PDF text extraction ────────────────────────────────────────────────────

def extract_text(pdf_path) -> str:
    path = str(pdf_path)

    # Try pdfplumber first — handles more complex layouts
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass

    # Fallback to pypdf
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        return ""


# ── Chunking ───────────────────────────────────────────────────────────────

def split_into_chunks(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks


# ── Endee connection ───────────────────────────────────────────────────────

def get_index():
    client = Endee()
    client.set_base_url(f"{ENDEE_URL}/api/v1")
    if ENDEE_TOKEN:
        client.set_auth_token(ENDEE_TOKEN)

    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )
        print(f"[Endee] Created index '{INDEX_NAME}'")
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print(f"[Endee] Using existing index '{INDEX_NAME}'")
        else:
            raise

    return client.get_index(name=INDEX_NAME)


# ── Single-file ingestion (used by Streamlit UI) ───────────────────────────

def ingest_pdf(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    model = SentenceTransformer(EMBED_MODEL)
    index = get_index()

    text = extract_text(path)
    if not text:
        raise ValueError(f"No extractable text in '{path.name}'.")

    chunks = split_into_chunks(text)
    vectors = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

    items = [
        {
            "id": str(uuid.uuid4()),
            "vector": vec.tolist(),
            "meta": {"text": chunk, "filename": path.name, "chunk_index": i},
        }
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]

    index.upsert(items)
    return {"filename": path.name, "chunks_ingested": len(items)}


# ── Bulk ingestion from data/ folder (CLI) ─────────────────────────────────

def ingest_all():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in '{DATA_DIR}'.")
        return

    print(f"[Info] Found {len(pdf_files)} PDF file(s)")
    total = 0
    for pdf_path in pdf_files:
        print(f"\n── Processing: {pdf_path.name}")
        result = ingest_pdf(str(pdf_path))
        print(f"   Stored: {result['chunks_ingested']} vectors ✓")
        total += result["chunks_ingested"]

    print(f"\n✅ Done — {total} total chunks stored in Endee.")


if __name__ == "__main__":
    ingest_all()
