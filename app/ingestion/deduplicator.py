import hashlib
import json
from pathlib import Path

def compute_document_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def load_seen_ids(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    with open(path) as f:
        for line in f:
            try:
                seen.add(json.loads(line)["document_id"])
            except Exception:
                continue
    return seen

def deduplicate(articles: list[dict], seen_ids: set) -> list[dict]:
    unique = []
    for a in articles:
        doc_id = compute_document_id(a["url"])
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            a["document_id"] = doc_id
            unique.append(a)
    return unique

def append_to_jsonl(articles: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")
