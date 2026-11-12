"""Shared RAG helpers: embeddings, FAISS vector store, retrieval, answers.

The demo works with NO paid API key. When no LLM is configured, answers are
extractive: the best-matching sentences from the retrieved chunks, clearly
labelled as such. When ``OPENAI_API_KEY`` is set, an optional LLM step can
turn the same retrieved chunks into a short grounded summary (see
``generate_answer``) — the retrieval and citations are identical either way.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# LangChain vector store (FAISS). The Hugging Face embeddings import is
# lazy (inside get_embeddings) so the demo runs without torch installed.
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
INDEX_DIR = BASE_DIR / "data" / "index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Words too common to use when picking the most relevant sentences.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "when", "where",
    "who", "why", "how", "do", "does", "did", "can", "could", "should",
    "i", "me", "my", "we", "you", "your", "it", "its", "this", "that",
    "and", "or", "of", "to", "in", "on", "for", "with", "by", "from",
    "about", "tell", "please", "give", "show", "much", "many", "long",
}


def get_embeddings():
    """Return the embedding model for the demo.

    Preferred: a local sentence-transformer model (``all-MiniLM-L6-v2``),
    which gives semantic (meaning-based) similarity and needs no API key.
    Fallback: a TF-IDF embedder (scikit-learn) when sentence-transformers
    is not installed — keyword-based, but the rest of the pipeline
    (chunking, FAISS search, grounded answers, citations) is identical,
    so the demo still runs end to end. The fallback is used automatically
    and reported by :func:`embedding_kind`.
    """
    try:
        import sentence_transformers  # noqa: F401  (model loads lazily)
        import torch  # noqa: F401
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception:
        return TfidfEmbeddings()


def embedding_kind(embeddings) -> str:
    """Human-readable name of the embedding model in use."""
    if isinstance(embeddings, TfidfEmbeddings):
        return "tf-idf (scikit-learn fallback)"
    return f"sentence-transformers/{EMBEDDING_MODEL}"


class TfidfEmbeddings(Embeddings):
    """Keyword-based embeddings (TF-IDF) with the same interface as
    LangChain's embedding models (``embed_documents`` / ``embed_query``).

    Used only when sentence-transformers is unavailable. The vectorizer is
    fitted on the documents at index-build time and re-loaded from
    ``data/index/tfidf.pkl`` at query time, so query vectors live in the
    same space as the indexed vectors.
    """

    _VECTORIZER_FILE = "tfidf.pkl"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer: TfidfVectorizer | None = None
        self._path = INDEX_DIR / self._VECTORIZER_FILE

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        matrix = self._vectorizer.fit_transform(texts)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        import pickle

        with open(self._path, "wb") as f:
            pickle.dump(self._vectorizer, f)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        import pickle

        if self._vectorizer is None:
            with open(self._path, "rb") as f:
                self._vectorizer = pickle.load(f)
        return self._vectorizer.transform([text]).toarray()[0].tolist()


def load_documents() -> list:
    """Load every ``*.md`` file under data/docs as one LangChain Document each."""
    from langchain_core.documents import Document

    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append(Document(page_content=path.read_text(encoding="utf-8"),
                             metadata={"source": path.name}))
    return docs


def chunk_documents(docs: list) -> list:
    """Split docs into overlapping chunks so retrieval returns short passages."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def build_index() -> FAISS:
    """Build a FAISS index from the sample docs and save it to data/index/."""
    chunks = chunk_documents(load_documents())
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def load_index() -> FAISS:
    """Load the saved FAISS index. Raises FileNotFoundError if not built yet."""
    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            "Vector index not found. Run: python -m app.ingest")
    return FAISS.load_local(
        str(INDEX_DIR), get_embeddings(),
        allow_dangerous_deserialization=True)  # index was built locally by us


def retrieve(vectorstore: FAISS, question: str, top_k: int = 3) -> list[dict]:
    """Return the top-k chunks for a question, each with source + score."""
    results = vectorstore.similarity_search_with_score(question, k=top_k)
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
            "score": round(float(score), 4),
        }
        for doc, score in results
    ]


def _keywords(question: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", question.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def extractive_answer(question: str, chunks: list[dict]) -> str:
    """Build a grounded answer from the retrieved chunks only.

    Picks the sentences that share the most keywords with the question.
    Never invents facts — if nothing matches, it says so.
    """
    keywords = _keywords(question)
    picked: list[str] = []
    # Walk chunks in retrieval order (best chunk first) and take the
    # most keyword-relevant sentences from each, up to 3 in total.
    # Markdown is cleaned first (headings dropped, newlines joined,
    # bold markers removed) so sentences are never cut mid-sentence
    # and the answer reads like plain text.
    # Scoring: a sentence scores for the question keywords it contains,
    # plus a small boost when its section heading matches the question
    # (e.g. the "height range" question matches the "What is the height
    # range?" heading, even though the answer sentence itself says
    # "The desk adjusts electrically from 62 cm...").
    for chunk in chunks:
        text = chunk["content"]
        headings = [ln for ln in text.splitlines()
                    if ln.strip().startswith("#")]
        heading_words = set(re.findall(
            r"[a-z0-9]+", " ".join(headings).lower()))
        heading_boost = 0.5 * len(keywords & heading_words)
        lines = [ln for ln in text.splitlines()
                 if not ln.strip().startswith("#")]
        clean = re.sub(r"\s+", " ", " ".join(lines))
        clean = re.sub(r"[*`_]+", "", clean).strip()
        scored: list[tuple[float, str]] = []
        for sentence in re.split(r"(?<=[.!?])\s+", clean):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            # Skip the sample-data disclaimer line ("...is a fictional
            # company/product used only as sample data...") — it is
            # metadata about the demo, not an answer to any question.
            lowered = sentence.lower()
            if "fictional" in lowered or "sample data" in lowered:
                continue
            words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            overlap = len(keywords & words)
            if overlap or heading_boost:
                scored.append((overlap + heading_boost, sentence))
        scored.sort(key=lambda item: -item[0])
        for _, sentence in scored:
            if sentence not in picked:
                picked.append(sentence)
            if len(picked) == 3:
                break
        if len(picked) == 3:
            break
    if not picked:
        return ("I could not find an answer to that question in the sample "
                "documents. Try asking about returns, shipping, the AeroDesk "
                "Pro, or company policies.")
    return " ".join(picked)


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Return (answer, mode) for a question and its retrieved chunks.

    Mode is "llm" only when OPENAI_API_KEY is set; otherwise "extractive".
    The LLM path is deliberately thin: it summarises the retrieved chunks
    and is instructed to say it does not know when the chunks do not cover
    the question, so it cannot hallucinate beyond the sources.
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            context = "\n\n".join(
                f"[{c['source']}] {c['content']}" for c in chunks)
            client = OpenAI()
            reply = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "Answer only from the context below. "
                        "If the context does not contain the answer, say so. "
                        "Keep the answer to 3 sentences.")},
                    {"role": "user", "content": (
                        f"Context:\n{context}\n\nQuestion: {question}")},
                ],
            )
            return reply.choices[0].message.content.strip(), "llm"
        except Exception:
            # Any LLM failure falls back to the grounded extractive answer.
            pass
    return extractive_answer(question, chunks), "extractive"
