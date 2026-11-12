"""FastAPI app: a RAG chatbot demo over the sample documents.

Endpoints:
    GET  /        — the web UI (static/index.html)
    GET  /health — readiness check; reports whether the index is loaded
    POST /ask    — {"question": "...", "top_k": 3}
                   -> {"answer", "mode", "sources": [{source, content, score}]}

Run:
    python -m app.ingest   # build the index first (once)
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.rag import BASE_DIR, generate_answer, load_index, retrieve

vectorstore = None  # loaded once at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    try:
        vectorstore = load_index()
    except FileNotFoundError:
        # The app still starts so /health can explain what to do.
        vectorstore = None
    yield


app = FastAPI(
    title="genai-rag-demo",
    description="A RAG chatbot demo: grounded answers with source citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


class Source(BaseModel):
    source: str
    content: str
    score: float


class AskResponse(BaseModel):
    answer: str
    mode: str  # "extractive" (no LLM) or "llm"
    sources: list[Source]


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": vectorstore is not None}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="Vector index not built yet. Run: python -m app.ingest",
        )
    chunks = retrieve(vectorstore, payload.question, top_k=payload.top_k)
    answer, mode = generate_answer(payload.question, chunks)
    return AskResponse(
        answer=answer,
        mode=mode,
        sources=[Source(**c) for c in chunks],
    )
