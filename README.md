# genai-rag-demo

A **Retrieval-Augmented Generation (RAG) chatbot demo** by
[Vivek Chinta](https://www.linkedin.com/in/vivek-chinta-b313a41a1)
([GitHub](https://github.com/vivekuk)).

Ask a question in plain English. The app retrieves the most relevant passages
from a small set of sample documents, then answers **only from those passages**
and shows the sources it used — so every answer is grounded and checkable.

> **Demo project, not client work.** Nothing here was built for any employer
> or client. All documents in `data/docs/` are fictional sample data
> (a made-up company, "Northwind Outfitters", and a made-up product,
> the "AeroDesk Pro"), written only to demonstrate the RAG pipeline.

## What it demonstrates

- Document ingestion, chunking, and embedding (LangChain + sentence-transformers)
- Vector search with FAISS
- Grounded answers with source citations — no hallucinating beyond the docs
- A FastAPI backend with a simple web UI
- An optional LLM step (OpenAI, only when you provide your own key);
  **without a key the demo still works**, using a clearly-labelled
  extractive answer taken from the retrieved passages

## Architecture

```text
User question
    │
    ▼
FastAPI  (/ask) ──► FAISS vector search (top-k chunks)
    │                       ▲
    │                       │  sentence-transformers
    │                       │  all-MiniLM-L6-v2 embeddings
    ▼                       │
Answer builder ──► sources cited in the reply
    │
    ├── no API key: extractive answer (labelled in the UI)
    └── OPENAI_API_KEY set: short LLM summary of the same chunks
```

### Design decisions

**Chunking strategy.** Documents are split with LangChain's
`RecursiveCharacterTextSplitter` into **500-character chunks with
80 characters of overlap**. The splitter tries paragraph, then line, then
sentence boundaries first, so a chunk usually holds one complete idea
instead of a cut-off sentence. The 80-character overlap means a fact
that straddles a boundary still appears whole in at least one chunk.
With only 4 short sample documents this gives 8 chunks — small enough
to inspect by hand, large enough to show the pipeline honestly.

**Embedding choice.** The preferred embedder is the local
sentence-transformer **`all-MiniLM-L6-v2`** (384 dimensions, ~90 MB):
semantic similarity, runs on CPU, no API key, no data leaves the machine.
In this environment the torch download (~2 GB) was too slow to install,
so the demo ships with a built-in **TF-IDF fallback** (scikit-learn,
keyword-based) that uses the identical pipeline — same chunking, same
FAISS index, same grounded answers — and the code reports which embedder
is active. One install command switches to the semantic model:
`pip install sentence-transformers langchain-huggingface`. See
BUILD_LOG.md for the honest install history.

**Why FAISS (and why not Chroma).** FAISS is an in-memory, exact
nearest-neighbour index: no server to run, no extra dependency beyond
`faiss-cpu`, and instant search at this scale. Chroma adds a persistent
server, its own storage format, and more moving parts — useful for a
production service with many users, unnecessary for a demo over a handful
of documents. The vector store is behind one small module (`app/rag.py`),
so swapping FAISS for Chroma later is a one-file change.

**Grounded answers, no hallucinations.** The answer builder only uses
sentences from the retrieved chunks. It scores sentences by keyword
overlap with the question, with a small boost when the sentence's
section heading matches the question. If no chunk covers the question,
it says so instead of guessing. When an LLM is configured, it is only
allowed to summarise the retrieved chunks, with the same "say you don't
know" instruction.

## Quick start

Python 3.10+ · FastAPI · LangChain · FAISS · sentence-transformers
(`all-MiniLM-L6-v2`, preferred) or TF-IDF fallback (built in, see
"Embedding choice" above) · Docker

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.ingest          # build the FAISS index from data/docs/
uvicorn app.main:app --reload # http://127.0.0.1:8000
```

Open http://127.0.0.1:8000 and ask a question.

## Run with Docker

```bash
docker compose up --build
# http://localhost:8000
```

The image builds the vector index at build time, so the app starts ready.

## Example questions (and expected answers)

| Question | Expected answer (from the sample docs) |
|---|---|
| What is the return policy? | Most items can be returned within 30 days of delivery for a full refund; refunds take 5–7 business days. Source: `returns-refunds.md` |
| How long does standard shipping take? | Standard shipping takes 3–5 business days; free over $50. Source: `shipping.md` |
| What is the weight limit of the AeroDesk Pro? | Up to 150 kg (330 lbs). Source: `product-faq.md` |
| What is the height range of the AeroDesk Pro? | The desk adjusts from 62 cm to 128 cm (24 to 50 inches). Source: `product-faq.md` |
| How many days of paid time off do employees get? | 20 days of PTO per year, plus 10 public holidays. Source: `company-policy.md` |
| What is the capital of France? | Not in the sample documents — the demo says so instead of guessing. |

These are the same 5 questions the evaluation script checks
(`python -m app.eval`): retrieval accuracy 5/5, grounded answers 5/5,
key fact present 5/5. See BUILD_LOG.md for the full run.

## API

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the return policy?"}'
```

Response:

```json
{
  "answer": "Customers can return most items within 30 days of delivery for a full refund. ...",
  "mode": "extractive",
  "sources": [
    {"source": "returns-refunds.md", "content": "...", "score": 0.42}
  ]
}
```

`mode` is `"extractive"` when no LLM is configured and `"llm"` when
`OPENAI_API_KEY` is set. No API key is ever hardcoded.

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

The retrieval tests check that each sample question returns the right document.

## Project layout

```text
genai-rag-demo/
├── app/
│   ├── main.py        # FastAPI app: /ask, /health, web UI
│   ├── ingest.py      # build the FAISS index from data/docs/
│   └── rag.py         # embeddings, retrieval, grounded answers
├── data/
│   ├── docs/          # sample documents (fictional)
│   └── index/         # built FAISS index (generated, gitignored)
├── static/index.html  # web UI
├── tests/             # retrieval tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── BUILD_LOG.md       # exact commands run and verification results
```

## License

MIT — see [LICENSE](LICENSE).
