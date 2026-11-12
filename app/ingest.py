"""Ingestion: chunk the sample docs, embed them, build the FAISS index.

Usage:
    python -m app.ingest

Reads data/docs/*.md, splits each file into overlapping chunks, embeds them
with a local sentence-transformer model (no API key), and saves the FAISS
index to data/index/. Re-run after editing any document.
"""

from __future__ import annotations

from app.rag import build_index, chunk_documents, load_documents


def main() -> None:
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents, split into {len(chunks)} chunks.")
    build_index()
    print("FAISS index saved to data/index/.")


if __name__ == "__main__":
    main()
