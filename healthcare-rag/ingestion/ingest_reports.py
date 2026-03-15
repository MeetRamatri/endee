

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

def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


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

    existing = client.list_indexes()  # returns list of index name strings
    if INDEX_NAME not in existing:
        print(f"[Endee] Creating index '{INDEX_NAME}' …")
        client.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )
    else:
        print(f"[Endee] Using existing index '{INDEX_NAME}'")

    return client.get_index(name=INDEX_NAME)


# ── Main pipeline ──────────────────────────────────────────────────────────

def ingest_all():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in '{DATA_DIR}'. Add reports and re-run.")
        return

    print(f"[Info] Found {len(pdf_files)} PDF file(s) in '{DATA_DIR}'")
    print(f"[Info] Loading embedding model: {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)

    index = get_index()
    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"\n── Processing: {pdf_path.name}")

        # 1. Extract text
        text = extract_text(pdf_path)
        if not text:
            print(f"   [!] No extractable text — skipping.")
            continue

        # 2. Split into chunks
        chunks = split_into_chunks(text)
        print(f"   Chunks  : {len(chunks)}")

        # 3. Embed all chunks at once (batched)
        vectors = model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # 4. Build upsert payload
        items = [
            {
                "id": str(uuid.uuid4()),
                "vector": vec.tolist(),
                "meta": {
                    "text": chunk,
                    "filename": pdf_path.name,
                    "chunk_index": i,
                },
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]

        # 5. Upsert to Endee
        index.upsert(items)
        print(f"   Stored  : {len(items)} vectors  ✓")
        total_chunks += len(items)

    print(f"\n✅ Ingestion complete — {total_chunks} total chunks stored in Endee.")


if __name__ == "__main__":
    ingest_all()
