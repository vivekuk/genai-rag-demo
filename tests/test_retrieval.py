"""Retrieval tests: the right document must win for each sample question.

Run:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import build_index, retrieve


def test_retrieval_returns_refund_policy():
    vectorstore = build_index()
    results = retrieve(vectorstore, "What is the return policy?", top_k=3)
    assert results, "retrieval returned no chunks"
    assert results[0]["source"] == "returns-refunds.md"


def test_retrieval_returns_shipping():
    vectorstore = build_index()
    results = retrieve(vectorstore, "How long does standard shipping take?", top_k=3)
    assert results, "retrieval returned no chunks"
    assert results[0]["source"] == "shipping.md"


def test_retrieval_returns_product_faq():
    vectorstore = build_index()
    results = retrieve(vectorstore, "What is the weight limit of the AeroDesk Pro?", top_k=3)
    assert results, "retrieval returned no chunks"
    assert results[0]["source"] == "product-faq.md"
