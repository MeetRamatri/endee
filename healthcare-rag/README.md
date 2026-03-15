# 🏥 Healthcare AI RAG System

> ⚠️ **Disclaimer:** This system is for **educational and demonstration purposes only**. It does **not** provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

## Overview

Healthcare professionals are often overwhelmed by dense medical literature, patient records, and clinical guidelines. Finding a specific piece of information inside a 300-page PDF typically means manual reading or keyword search — both are slow and error-prone.

This project solves that by building a **Healthcare AI RAG (Retrieval-Augmented Generation) System** that:

- **Ingests** medical PDF reports and stores them as semantic vector embeddings
- **Retrieves** the most relevant document sections for any natural-language question
- **Generates** a clear, grounded answer using Google Gemini — based *only* on the uploaded documents, not on hallucinated knowledge

---

## Problem Statement

| Challenge | This System's Solution |
|---|---|
| Medical PDFs are long and hard to navigate | Split into searchable 400-word chunks |
| Keyword search misses semantically related content | Vector similarity search finds conceptually relevant chunks |
| LLMs hallucinate medical information | RAG grounds the answer strictly in the uploaded document |
| Uploading documents to cloud LLMs raises privacy concerns | Embeddings stay local in Endee; only the retrieved excerpt is sent to Gemini |

---

## System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                           │
│                                                                     │
│  PDF File  ──►  pdfplumber / pypdf  ──►  Text Chunks (400 words)   │
│                                                  │                  │
│                           sentence-transformers  ▼                  │
│                         (all-MiniLM-L6-v2)  Embeddings             │
│                                                  │                  │
│                                         Endee Vector DB ◄───────── │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          RAG QUERY PIPELINE                         │
│                                                                     │
│  User Question  ──►  Embedding  ──►  Endee Semantic Search          │
│                                              │                      │
│                                    Top-3 Relevant Chunks            │
│                                              │                      │
│                                    Prompt Template                  │
│                                              │                      │
│                                    Google Gemini  ──►  Answer       │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Module | File | Responsibility |
|---|---|---|
| Ingestion | `ingestion/ingest_reports.py` | PDF → chunks → vectors → Endee |
| Query Engine | `rag/query_engine.py` | Question → retrieval → Gemini generation |
| Web UI | `ui/app.py` | Streamlit interface for upload, extraction & Q&A |

---

## How Endee Is Used

[Endee](https://endee.io) is a lightweight, self-hosted vector database designed for fast semantic search. This project uses Endee's Python SDK for three operations:

### 1. Index Creation
```python
client.create_index(
    name="medical_reports",
    dimension=384,        # matches all-MiniLM-L6-v2 output size
    space_type="cosine",  # cosine similarity for normalized embeddings
    precision=Precision.INT8,  # quantized for efficiency
)
```

### 2. Upserting Vectors (Ingestion)
Each document chunk is stored with its embedding and metadata:
```python
index.upsert([{
    "id": "<uuid>",
    "vector": [0.12, -0.04, ...],   # 384-dimensional float list
    "meta": {
        "text": "Patient was prescribed...",
        "filename": "report.pdf",
        "chunk_index": 7,
    }
}])
```

### 3. Semantic Search (Retrieval)
At query time, the user's question is embedded and the top-3 closest chunks are fetched:
```python
results = index.query(vector=question_embedding, top_k=3)
# Returns: [{id, similarity, meta: {text, filename, chunk_index}}]
```

The retrieved `text` fields are assembled into a context block and passed to Gemini with the original question.

---

## Project Structure

```
healthcare-rag/
├── data/                        # Place PDF reports here for bulk ingestion
├── ingestion/
│   ├── __init__.py
│   └── ingest_reports.py        # extract_text, split_into_chunks, ingest_pdf, ingest_all
├── rag/
│   ├── __init__.py
│   └── query_engine.py          # embed_query, vector_search, build_context, rag
├── ui/
│   └── app.py                   # Streamlit web app
├── .env.example                 # Environment variable template
├── requirements.txt
└── README.md
```

---

## Setup & Execution

### Prerequisites

- **Docker** — to run the Endee vector database
- **Python 3.11+**
- **Gemini API Key** — free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

### Step 1 — Start Endee

```bash
docker run --ulimit nofile=100000:100000 -p 8080:8080 \
  -v ./endee-data:/data endeeai/endee:latest
```

Endee is now running at `http://localhost:8080`.

---

### Step 2 — Create Virtual Environment & Install Dependencies

```bash
cd healthcare-rag
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 3 — Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
ENDEE_URL=http://localhost:8080
ENDEE_INDEX_NAME=medical_reports
```

---

### Step 4 — Run the Web UI

```bash
streamlit run ui/app.py
```

Open **http://localhost:8501** in your browser.

**Workflow:**
1. Upload a PDF in the sidebar → extracted text is shown for verification
2. Click **⚡ Ingest into Endee** → chunks are embedded and stored
3. Type a question → AI retrieves relevant sections and generates an answer

---

### Alternative: CLI Usage

```bash
# Ingest all PDFs from the data/ folder at once
python ingestion/ingest_reports.py

# Ask a question from the terminal
python rag/query_engine.py "What was the patient's diagnosis?"
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `ENDEE_URL` | `http://localhost:8080` | Endee server address |
| `ENDEE_AUTH_TOKEN` | *(empty)* | Auth token if Endee auth is enabled |
| `ENDEE_INDEX_NAME` | `medical_reports` | Name of the vector index |
| `GEMINI_API_KEY` | — | **Required.** Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for generation |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector Database | [Endee](https://endee.io) (self-hosted) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| PDF Extraction | `pdfplumber` (primary) + `pypdf` (fallback) |
| LLM | Google Gemini via `google-genai` SDK |
| Web UI | Streamlit |
| Language | Python 3.11 |
