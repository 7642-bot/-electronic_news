import json
import re
from pathlib import Path
from app.config.settings import PATHS
from app.ingestion.dictionary_loader import load_dictionaries

def _extract_entries_from_text(text: str) -> list[dict]:
    """Heuristic extraction of term/definition pairs from raw dictionary pages.
    Adjust regex for your PDFs if needed."""
    entries = []
    # Match patterns like: "Term: definition" or "Term — definition"
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"^([A-Z][\w\s\-/()]{2,60}?)\s*[:\-—]\s*(.{20,400})$", line)
        if m:
            entries.append({"term": m.group(1).strip(), "definition": m.group(2).strip()})
    return entries

def build():
    """Build instruction dataset from all dictionary PDFs."""
    print("Loading dictionaries from", PATHS["dataset"])
    pages = load_dictionaries(PATHS["dataset"])

    entries = []
    for p in pages:
        entries.extend(_extract_entries_from_text(p.page_content))
    print(f"Extracted {len(entries)} term/definition candidates.")

    samples = []
    for e in entries:
        term, definition = e["term"], e["definition"]
        samples.append({
            "text": f"### Instruction:\nDefine '{term}' in electronics engineering.\n\n### Response:\n{definition}"
        })
        samples.append({
            "text": f"### Instruction:\nExplain '{term}' in a technical context.\n\n### Response:\n{definition} This term is commonly used in electronics and communication engineering."
        })

    out = PATHS["finetune"] / "electronics_qa.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"✓ Wrote {len(samples)} training samples to {out}")
    return out
