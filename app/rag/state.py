from typing import TypedDict

class RAGState(TypedDict, total=False):
    query: str
    retrieved_docs: list[dict]
    dictionary_context: list[str]
    summary: str
    sources: list[str]
    quality_score: float
    needs_review: bool
