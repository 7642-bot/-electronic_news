import feedparser
from datetime import datetime
from app.config.settings import FEEDS

def fetch_feed(feed_url: str) -> list[dict]:
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries:
        articles.append({
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "summary": entry.get("summary", ""),
            "published": entry.get("published", datetime.utcnow().isoformat()),
            "source": feed.feed.get("title", ""),
        })
    return articles

def fetch_all() -> list[dict]:
    all_articles = []
    for feed in FEEDS:
        if not feed.enabled:
            continue
        try:
            articles = fetch_feed(feed.feed_url)
            for a in articles:
                a["source_id"] = feed.source_id
            all_articles.extend(articles)
            print(f"[{feed.source_name}] {len(articles)} articles")
        except Exception as e:
            print(f"[{feed.source_name}] FAILED: {e}")
    return all_articles
