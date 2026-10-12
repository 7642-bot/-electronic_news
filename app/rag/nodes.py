import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from app.config.settings import EMBEDDING_MODEL, PATHS, NEWS_COLLECTION, DICT_COLLECTION, LLM_MODEL
from app.rag.state import RAGState

_emb = None
_news_store = None
_dict_store = None
_llm = None

def _get_emb():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _emb

def _get_news_store():
    global _news_store
    if _news_store is None:
        _news_store = Chroma(
            persist_directory=str(PATHS["chroma"]),
            embedding_function=_get_emb(),
            collection_name=NEWS_COLLECTION,
        )
    return _news_store

def _get_dict_store():
    global _dict_store
    if _dict_store is None:
        _dict_store = Chroma(
            persist_directory=str(PATHS["chroma"]),
            embedding_function=_get_emb(),
            collection_name=DICT_COLLECTION,
        )
    return _dict_store

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=LLM_MODEL, temperature=0.2)
    return _llm

def retrieve_news(state: RAGState) -> RAGState:
    docs = _get_news_store().similarity_search(state["query"], k=5)
    state["retrieved_docs"] = [
        {"content": d.page_content, "source": d.metadata.get("source", "")}
        for d in docs
    ]
    state["sources"] = list({d["source"] for d in state["retrieved_docs"]})
    return state

def lookup_dictionary(state: RAGState) -> RAGState:
    docs = _get_dict_store().similarity_search(state["query"], k=3)
    state["dictionary_context"] = [d.page_content for d in docs]
    return state

def generate_summary(state: RAGState) -> RAGState:
    news = "\n\n".join(f"[{d['source']}]: {d['content']}" for d in state.get("retrieved_docs", []))
    dic = "\n".join(state.get("dictionary_context", []))
    prompt = f"""You are an electronics technology intelligence analyst.
Generate a concise, technically accurate summary using the context below.

DICTIONARY CONCEPTS:
{dic}

NEWS CONTEXT:
{news}

QUERY: {state['query']}

Write a 3-5 sentence technical summary using correct electronics terminology."""
    resp = _get_llm().invoke(prompt)
    state["summary"] = resp.content
    return state

def verify_quality(state: RAGState) -> RAGState:
    try:
        resp = _get_llm().invoke(
            f"Rate technical accuracy (1-10, reply with just the number):\n\n{state['summary']}"
        )
        score = float(resp.content.strip().split()[0])
    except Exception:
        score = 5.0
    state["quality_score"] = score
    state["needs_review"] = score < 6.0
    return state

def route_query(state: RAGState) -> str:
    q = state["query"].lower()
    if any(k in q for k in ["news", "recent", "latest", "update", "report"]):
        return "retrieve"
    return "direct"
