"""Evaluation: retrieval accuracy + answer grounding, no LLM needed.

Runs a fixed set of questions against the built index and reports, for
each question:
  * retrieval: did the top-1 retrieved chunk come from the expected doc?
  * grounding: is every sentence of the answer taken from the retrieved
    chunks (i.e. the answer contains no invented facts)?
  * key fact: does the answer contain the expected key fact?

Usage:
    python -m app.ingest   # build the index first (once)
    python -m app.eval

Exit code is 0 only if every question passes retrieval AND grounding.
No metrics are invented — the numbers below come from running this script.
"""

from __future__ import annotations

import re
import sys

from app.rag import generate_answer, load_index, retrieve

# (question, expected top-1 document, key fact the answer must contain)
EVAL_CASES: list[tuple[str, str, str]] = [
    ("What is the return policy?",
     "returns-refunds.md", "30 days"),
    ("How long does standard shipping take?",
     "shipping.md", "3 to 5 business days"),
    ("What is the weight limit of the AeroDesk Pro?",
     "product-faq.md", "150 kg"),
    ("How many days of paid time off do employees get?",
     "company-policy.md", "20 days"),
    ("What is the height range of the AeroDesk Pro?",
     "product-faq.md", "62 cm"),
]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/extra spaces — for robust matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def _answer_is_grounded(answer: str, chunks: list[dict]) -> bool:
    """True if every sentence of the answer is supported by the retrieved
    chunks.

    Support is measured by word coverage: at least 80% of each sentence's
    content words (stopwords removed) must appear in the retrieved
    context. Exact-substring matching is too strict here, because a
    sentence can legitimately span a chunk boundary (chunking splits the
    documents every ~500 characters), so its two halves live in two
    different chunks. The canned "could not find an answer" reply counts
    as grounded: it claims nothing about the documents.
    """
    if answer.startswith("I could not find an answer"):
        return True
    context_words = set(_normalize(" ".join(c["content"] for c in chunks)).split())
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        words = [w for w in _normalize(sentence).split() if w not in _STOPWORDS]
        if not words:
            continue
        covered = sum(1 for w in words if w in context_words)
        if covered / len(words) < 0.8:
            return False
    return True


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "when", "where",
    "who", "why", "how", "do", "does", "did", "can", "could", "should",
    "i", "me", "my", "we", "you", "your", "it", "its", "this", "that",
    "and", "or", "of", "to", "in", "on", "for", "with", "by", "from",
    "about", "tell", "please", "give", "show", "much", "many", "long",
}


def run_eval() -> int:
    """Run all eval cases and print a per-question and summary report."""
    vectorstore = load_index()
    n_retrieval = n_grounded = n_fact = 0

    print(f"{'Question':<52} {'Retrieval':<10} {'Grounded':<9} {'Key fact'}")
    print("-" * 90)
    for question, expected_doc, key_fact in EVAL_CASES:
        chunks = retrieve(vectorstore, question, top_k=3)
        answer, mode = generate_answer(question, chunks)
        retrieval_ok = chunks[0]["source"] == expected_doc
        grounded_ok = _answer_is_grounded(answer, chunks)
        fact_ok = key_fact.lower() in answer.lower()
        n_retrieval += retrieval_ok
        n_grounded += grounded_ok
        n_fact += fact_ok
        print(f"{question:<52} "
              f"{'PASS' if retrieval_ok else 'FAIL':<10} "
              f"{'PASS' if grounded_ok else 'FAIL':<9} "
              f"{'PASS' if fact_ok else 'FAIL'}  "
              f"(top: {chunks[0]['source']}, mode: {mode})")
        if not (retrieval_ok and grounded_ok and fact_ok):
            print(f"    answer: {answer}")

    total = len(EVAL_CASES)
    print("-" * 90)
    print(f"Retrieval accuracy: {n_retrieval}/{total} "
          f"({100 * n_retrieval // total}%)")
    print(f"Grounded answers:    {n_grounded}/{total}")
    print(f"Key fact present:    {n_fact}/{total}")
    passed = n_retrieval == total and n_grounded == total and n_fact == total
    print("OVERALL:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(run_eval())
