from app.rag.graph import app

if __name__ == "__main__":
    queries = [
        "Latest advances in 6G terahertz communication",
        "What is impedance matching?",
        "Recent developments in GaN power electronics",
    ]
    for q in queries:
        print(f"\n{'='*60}\nQUERY: {q}\n{'='*60}")
        result = app.invoke({"query": q})
        print(result.get("summary", "(no summary)"))
        print(f"Sources: {result.get('sources', [])}")

# This serves as a demonstration of the RAG pipeline for the project. It imports the `app` object from the `rag.graph` module, which represents the RAG graph application. When run directly, it executes a series of predefined queries related to electronics technology and prints the generated summaries along with their sources. The script showcases how the RAG pipeline retrieves relevant context and generates informative summaries based on user queries.