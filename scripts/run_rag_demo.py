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
