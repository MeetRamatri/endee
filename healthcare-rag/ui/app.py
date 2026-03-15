import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Healthcare RAG",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Healthcare AI Assistant")
st.caption("Upload a medical report, then ask questions about it.")

st.warning(
    "⚠️ **Disclaimer:** This Healthcare AI system is for educational and demonstration "
    "purposes only. It does not provide medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional for medical concerns."
)

st.divider()

# ── Session state ──────────────────────────────────────────────────────────

if "ingested" not in st.session_state:
    st.session_state.ingested = []

# ── Sidebar — Upload, Extract & Ingest ────────────────────────────────────

with st.sidebar:
    st.header("📁 Upload Report")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        # Save to temp file for reading
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        # ── Extract & preview ──────────────────────────────────────────────
        try:
            from ingestion.ingest_reports import extract_text
            from pypdf import PdfReader

            reader   = PdfReader(tmp_path)
            num_pages = len(reader.pages)
            extracted = extract_text(tmp_path)
            word_count = len(extracted.split())
        except Exception as e:
            extracted, num_pages, word_count = "", 0, 0
            st.error(f"Could not read PDF: {e}")

        st.caption(f"📄 {num_pages} page(s) — {word_count} words extracted")

        with st.expander("🔍 View Extracted Text"):
            st.text_area(
                label="extracted",
                value=extracted if extracted else "No text found.",
                height=250,
                disabled=True,
                label_visibility="collapsed",
            )

        # ── Ingest ────────────────────────────────────────────────────────
        if not extracted:
            st.warning(
                "⚠️ No text could be extracted from this PDF. "
                "It may be a **scanned / image-based** document. "
                "Please upload a text-based PDF."
            )
        elif st.button("⚡ Ingest into Endee", use_container_width=True):
            if uploaded.name in st.session_state.ingested:
                st.info(f"'{uploaded.name}' is already ingested.")
            else:
                with st.spinner("Storing embeddings in Endee…"):
                    try:
                        from ingestion.ingest_reports import ingest_pdf
                        result = ingest_pdf(tmp_path)
                        st.session_state.ingested.append(uploaded.name)
                        st.success(
                            f"✅ Ingested **{result['chunks_ingested']}** chunks "
                            f"from *{result['filename']}*"
                        )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

        # Clean up temp file once we're done with the sidebar render
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    if st.session_state.ingested:
        st.divider()
        st.markdown("**Ingested documents:**")
        for name in st.session_state.ingested:
            st.markdown(f"- 📄 {name}")

# ── Main — Q&A ─────────────────────────────────────────────────────────────

question = st.text_input(
    "💬 Ask a question about your report",
    placeholder="e.g. What medications were prescribed?",
)

if st.button("Get Answer", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.ingested:
        st.warning("Please upload and ingest a report first.")
    else:
        with st.spinner("Searching documents and generating answer…"):
            try:
                from rag.query_engine import rag
                result = rag(question)

                st.divider()
                st.subheader("🤖 Answer")
                st.write(result["answer"])

                with st.expander(f"📚 Sources — {len(result['chunks'])} chunk(s) retrieved"):
                    for i, chunk in enumerate(result["chunks"], 1):
                        st.markdown(
                            f"**[{i}]** `{chunk['filename']}` — "
                            f"similarity: `{chunk['similarity']}`"
                        )
                        st.caption(chunk["text"][:300] + "…")
                        if i < len(result["chunks"]):
                            st.divider()

            except Exception as e:
                st.error(f"Error: {e}")
