import json
from datetime import datetime
from app.ingestion.rss_fetcher import fetch_all
from app.ingestion.deduplicator import deduplicate, load_seen_ids, append_to_jsonl, compute_document_id
from app.ingestion.vector_indexer import index_news, load_processed_articles
from app.config.settings import PATHS
from app.rag.graph import app as rag_app

def run_ingestion():
    print(f"[{datetime.utcnow()}] Ingesting news...")
    docs_path = PATHS["processed"] / "documents.jsonl"
    seen = load_seen_ids(docs_path)
    articles = fetch_all()
    new_articles = deduplicate(articles, seen)
    append_to_jsonl(new_articles, docs_path)
    print(f"Ingested {len(new_articles)} new articles.")
    return new_articles

def run_indexing():
    articles = load_processed_articles()
    if not articles:
        print("No articles to index.")
        return
    index_news(articles[-100:])

def run_summarization(limit: int = 10):
    articles = load_processed_articles()
    for a in articles[-limit:]:
        result = rag_app.invoke({"query": a["title"]})
        print(f"\n--- {a['title']} ---\n{result.get('summary', '')}\n")

def daily_pipeline():
    run_ingestion()
    run_indexing()
    run_summarization(limit=5)

if __name__ == "__main__":
    daily_pipeline()
