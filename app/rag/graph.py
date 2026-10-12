from langgraph.graph import StateGraph, START, END
from app.rag.state import RAGState
from app.rag.nodes import (
    retrieve_news, lookup_dictionary, generate_summary,
    verify_quality, route_query,
)

def build_graph():
    wf = StateGraph(RAGState)
    wf.add_node("retrieve_news", retrieve_news)
    wf.add_node("lookup_dictionary", lookup_dictionary)
    wf.add_node("generate_summary", generate_summary)
    wf.add_node("verify_quality", verify_quality)

    wf.add_conditional_edges(
        START, route_query,
        {"retrieve": "retrieve_news", "direct": "lookup_dictionary"},
    )
    wf.add_edge("retrieve_news", "lookup_dictionary")
    wf.add_edge("lookup_dictionary", "generate_summary")
    wf.add_edge("generate_summary", "verify_quality")
    wf.add_edge("verify_quality", END)
    return wf.compile()

app = build_graph()
