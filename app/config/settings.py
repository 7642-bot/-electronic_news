import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

@dataclass
class RSSSourceConfig:
    source_id: str
    source_name: str
    source_type: str
    feed_url: str
    enabled: bool = True

FEEDS = [
    RSSSourceConfig("ieee_spectrum", "IEEE Spectrum", "rss",
                    "https://spectrum.ieee.org/feeds/feed.rss"),
    RSSSourceConfig("eetimes", "EE Times", "rss",
                    "https://www.eetimes.com/feed/"),
    RSSSourceConfig("semiconductor_today", "Semiconductor Today", "rss",
                    "http://www.semiconductor-today.com/rss/news.xml"),
    RSSSourceConfig("electronicsforu", "Electronics For You", "rss",
                    "https://www.electronicsforu.com/feed"),
    RSSSourceConfig("techcrunch_ai", "TechCrunch AI", "rss",
                    "https://techcrunch.com/category/artificial-intelligence/feed/"),
    RSSSourceConfig("ars_technica", "Ars Technica", "rss",
                    "https://feeds.arstechnica.com/arstechnica/index"),
    RSSSourceConfig("mit_tech_review", "MIT Tech Review", "rss",
                    "http://feeds.technologyreview.com/technology_review_top_stories"),
    RSSSourceConfig("hacker_news", "Hacker News", "rss",
                    "https://news.ycombinator.com/rss"),
]

PATHS = {
    "dataset": BASE_DIR / os.getenv("DATASET_DIR", "dataset"),
    "raw": BASE_DIR / "data" / "raw",
    "processed": BASE_DIR / "data" / "processed",
    "finetune": BASE_DIR / "data" / "finetune",
    "chroma": BASE_DIR / os.getenv("CHROMA_DIR", "chroma_db"),
    "models": BASE_DIR / os.getenv("MODEL_DIR", "models"),
    "logs": BASE_DIR / "logs",
}

for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEWS_COLLECTION = "news_articles"
DICT_COLLECTION = "dictionaries"
SUMMARY_COLLECTION = "summaries"

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B")
