#  Electronics Technology Intelligence System

An AI-powered RAG pipeline that ingests electronics/technology news, indexes domain-specific dictionaries, and generates grounded technical summaries — with an interactive web demo and contextual-bandit personalization.

---

## Overview

This system solves a specific problem: **general-purpose LLMs don't know your domain**. They hallucinate specific values, miss current events, and use imprecise terminology.

The Electronics Technology Intelligence System fixes this by:

1. **Ingesting** electronics news from RSS feeds on demand
2. **Indexing** authoritative electronics dictionaries as a retrieval corpus
3. **Grounding** LLM summaries in retrieved dictionary definitions + recent news
4. **Routing** queries intelligently — definitions go to dictionaries, event queries go to news
5. **Personalizing** recommendations using a LinUCB contextual bandit
6. **Verifying** generated content with a self-rated quality gate

The result: summaries that cite real sources, use correct electronics terminology, and improve with user feedback.

---

##  What Actually Works 


| RSS ingestion             | ✅ Working | 8 feeds configured, 4 actively returning data |
| Deduplication             | ✅ Working | SHA-256 URL hashing — idempotent |
| PDF dictionary loading    | ✅ Working | PyMuPDFLoader, 5 dictionaries loaded |
| Chunking + embedding      | ✅ Working | 14,697 dict chunks + 850 news chunks |
| Vector storage            | ✅ Working | ChromaDB with persistent storage |
| Semantic retrieval        | ✅ Working | Cosine similarity via all-MiniLM-L6-v2 |
| LangGraph orchestration   | ✅ Working | Conditional routing between news/dict paths |
| Groq LLM generation       | ✅ Working | `openai/gpt-oss-120b` |
| Quality verification      | ✅ Working | Self-rated 1–10 by separate LLM call |
| Streamlit demo UI         | ✅ Working | 4 tabs: Query, Explore, Bandit, Feeds |
| LinUCB contextual bandit  | ✅ Working | Standalone; UI demo included |
| Fine-tuning               | 🟡 Scaffolding | `dataset_builder.py` exists, not run |
| InShorts-style summarizer | 🔴 Planned | Future work |
| Mass web crawling         | 🔴 Planned | Future work |

**Overall completion: ~45%** — functional MVP, not production-ready.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ORCHESTRATOR                       │
│   route_query: "news"-type queries → retrieve_news              │
│                definition queries   → lookup_dictionary         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │  NEWS RETRIEVER │         │  DICT RETRIEVER │
    │  ChromaDB       │         │  ChromaDB       │
    │  news_articles  │         │  dictionaries   │
    └────────┬────────┘         └────────┬────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
              ┌────────────────────────┐
              │   PROMPT ASSEMBLY      │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │   GROQ LLM             │
              │   gpt-oss-120b         │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │   QUALITY VERIFIER     │
              │   (LLM-as-judge)       │
              └────────────┬───────────┘
                           ▼
                     SUMMARY + SOURCES

    ──────── OFFLINE / PARALLEL ────────

    User clicks → LinUCB bandit → preference model → smarter next recommendation
```

### Data Flow

```
RSS feeds ──► feedparser ──► dedup (SHA-256) ──► documents.jsonl
                                                       │
                                                       ▼
PDFs ──► PyMuPDFLoader ──► splitter ──► embedder ──► ChromaDB
                                                       │
                                                       ▼
                                              LangGraph RAG
                                                       │
                                                       ▼
                                                 Summary + UI
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- pip and venv
- A [Groq API key](https://console.groq.com) (free tier available)

### Installation

```bash
# 1. Clone / create the project
cd electronics-intel

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install core dependencies
python -m pip install python-dotenv feedparser requests chromadb sentence-transformers \
    langchain langchain-community langchain-groq langchain-text-splitters \
    langchain-huggingface langchain-chroma \
    pypdf pymupdf pdfplumber langgraph numpy apscheduler tqdm streamlit pytest

# 5. Install project as editable package
python -m pip install -e .
```

> ** Windows without CUDA GPU:** Skip `bitsandbytes`, `transformers`, `peft`, `trl`, `accelerate`, and `torch` for now — they're only needed for fine-tuning.

### Configure

```bash
cp .env.example .env
```

Edit `.env` and add your Groq key:

```bash
GROQ_API_KEY=gsk_your_actual_key_here
LLM_MODEL=openai/gpt-oss-120b
```

> ** Note:** Groq decommissioned `llama-3.3-70b-versatile` in August 2026. Use `openai/gpt-oss-120b` (Production tier) or `openai/gpt-oss-20b` (cheaper).

### Add Your Dictionaries

Place all electronics dictionary PDFs here:

```
electronics-intel/dataset/
├── your-dictionary-1.pdf
├── your-dictionary-2.pdf
└── ...
```

The loader scans this folder for `*.pdf` files (non-recursive).

### Build the Index

```bash
# 1. Fetch news from configured RSS feeds
python scripts/run_ingestion.py

# 2. Index news + dictionaries into ChromaDB
#    First run downloads the embedding model (~90 MB) and takes ~20 min for large PDFs.
#    Embeddings are cached to chroma_db/dict_embeddings.npy — subsequent runs are instant.
python scripts/run_indexing.py
```

### Run the Demo

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## Using the Demo UI

### Tab 1: Query

The main RAG interface. Type a question or click an example.

**What you'll see:**
- Generated summary (grounded in retrieved context)
- Metrics: number of sources, news chunks, dict chunks
- Expandable panels showing **exactly what was retrieved** — this is the transparency that proves grounding
- Quality warning if the self-rater scored below 6/10

**Routing behavior:**
- Queries containing "news", "latest", "recent", "update", "report" → both retrievers run
- All other queries → dictionary-only

### Tab 2: Explore Corpus

Direct semantic search with distance scores. **Use this to diagnose bad summaries.**

- Distance **< 0.8** = excellent match
- Distance **0.8–1.2** = relevant
- Distance **> 1.5** = weak/irrelevant (a signal that the corpus doesn't cover this topic)

### Tab 3: Bandit Demo

Interactive LinUCB simulator. Set the "true preference" arm, run N rounds, and watch the algorithm learn which arm to pick.

### Tab 4: Feeds

Trigger RSS ingestion from the browser. Shows:
- Ingested article count
- Latest articles in the corpus
- Total articles metric

---

## Project Structure

```
electronics-intel/
├── .env.example                    # Environment variable template
├── .gitignore
├── README.md                       # ← you are here
├── requirements.txt
├── setup.py                        # Enables `pip install -e .`
├── streamlit_app.py                # Demo UI
├── dataset/                        # ← PUT YOUR DICTIONARY PDFs HERE
├── data/
│   ├── raw/                        # Raw fetched content (unused)
│   ├── processed/
│   │   └── documents.jsonl         # Ingested articles (append-only)
│   └── finetune/                   # Training datasets (future)
├── chroma_db/                      # Persistent vector store
│   └── dict_embeddings.npy         # Embedding cache (~50 MB)
├── models/                         # Fine-tuned adapters (future)
├── logs/
├── app/
│   ├── config/
│   │   └── settings.py             # FEEDS, PATHS, model names
│   ├── ingestion/
│   │   ├── rss_fetcher.py          # feedparser wrapper
│   │   ├── deduplicator.py         # SHA-256 dedup
│   │   ├── dictionary_loader.py    # PyMuPDFLoader + chunker
│   │   └── vector_indexer.py       # ChromaDB indexing (batched)
│   ├── rag/
│   │   ├── state.py                # RAGState TypedDict
│   │   ├── nodes.py                # retrieve/lookup/generate/verify
│   │   └── graph.py                # LangGraph assembly
│   ├── bandit/
│   │   └── linucb.py               # Contextual bandit
│   ├── finetuning/                 # Future work
│   ├── pipeline/
│   │   └── daily_pipeline.py       # Orchestration
│   └── utils/
│       └── logger.py
├── scripts/
│   ├── run_ingestion.py
│   ├── run_indexing.py
│   ├── run_rag_demo.py
│   ├── run_finetune.py             # Future
│   └── run_scheduler.py
└── tests/
    └── test_smoke.py
```

---

##  Configuration

### RSS Feeds

Edit `app/config/settings.py`:

```python
FEEDS = [
    RSSSourceConfig("ieee_spectrum", "IEEE Spectrum", "rss",
                    "https://spectrum.ieee.org/feeds/feed.rss"),
    # Add more here — see " Sources" below
]
```

###  Sources


| IEEE Spectrum       | `https://spectrum.ieee.org/feeds/feed.rss` | Broad engineering |
| EE Times            | `https://www.eetimes.com/feed/` | Electronics/semiconductor |
| Semiconductor Today | `http://www.semiconductor-today.com/rss/news.xml` | Compound semis |
| Electronics For You | `https://www.electronicsforu.com/feed` | Embedded/design |
| TechCrunch AI       | `https://techcrunch.com/category/artificial-intelligence/feed/` | AI hardware |
| Ars Technica        | `https://feeds.arstechnica.com/arstechnica/index` | Deep tech |
| MIT Tech Review     | `http://feeds.technologyreview.com/technology_review_top_stories` | Long-form |
| Hacker News         | `https://news.ycombinator.com/rss` | Community signal |

> **Some feeds return 0 articles** — their URLs may have changed or they may block automated fetches. Non-blocking.

### Environment Variables

| `GROQ_API_KEY`      | Groq LLM access         | (required) |
| `LLM_MODEL`         | Groq model ID           | `openai/gpt-oss-120b` |
| `LANGSMITH_TRACING` | Enable LangSmith traces | `false` |
| `LANGSMITH_API_KEY` | LangSmith API key       | (optional) |
| `CHROMA_DIR`        | Vector store location   | `./chroma_db` |
| `BASE_MODEL`        | Base for fine-tuning    | `meta-llama/Llama-3.1-8B` |

---

##  Testing

### Smoke Test

```bash
python -m pytest tests/test_smoke.py -v -p no:langsmith
```

> **Note:** The `-p no:langsmith` flag disables the LangSmith pytest plugin, which can cause tests to be deselected.

### Manual Verification

**Verify collections:**

```bash
python -c "
import chromadb
from app.config.settings import PATHS
c = chromadb.PersistentClient(path=str(PATHS['chroma']))
print('News:', c.get_collection('news_articles').count())
print('Dict:', c.get_collection('dictionaries').count())
"
```

**Test retrieval:**

```bash
python -c "
from app.rag.nodes import _get_dict_store
for d in _get_dict_store().similarity_search('impedance matching', k=3):
    print('•', d.page_content[:200].replace(chr(10), ' '))
    print()
"
```

**Full RAG demo:**

```bash
python scripts/run_rag_demo.py
```

**Bandit simulation:**

```bash
python -c "
import numpy as np
from app.bandit.linucb import LinUCB
bandit = LinUCB(n_arms=3, context_dim=5)
ctx = np.random.randn(5, 1)
for _ in range(200):
    arm = bandit.select(ctx)
    bandit.update(arm, ctx, 1.0 if arm == 1 else 0.0)
print('Learned best arm:', bandit.select(ctx))  # Expected: 1
"
```

---

##  Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

**Fix:** Install as editable package from project root:

```bash
python -m pip install -e .
```

### `ModuleNotFoundError: No module named 'langchain.text_splitter'`

**Cause:** LangChain moved text splitters to a separate package.

**Fix:** Change imports in all `.py` files:

```python
# OLD
from langchain.text_splitter import RecursiveCharacterTextSplitter

# NEW
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

One-liner (PowerShell):

```powershell
Get-ChildItem -Recurse -Filter *.py app | ForEach-Object {
    (Get-Content $_.FullName) -replace 'from langchain\.text_splitter import', 'from langchain_text_splitters import' | Set-Content $_.FullName
}
```

### `groq.NotFoundError: model does not exist`

**Cause:** Groq decommissioned `llama-3.3-70b-versatile`.

**Fix:** Update `.env`:

```
LLM_MODEL=openai/gpt-oss-120b
```

### `chromadb.errors.InternalError: Batch size of X is greater than max batch size of 5461`

**Cause:** ChromaDB has a 5,461-item limit per `add()` call.

**Fix:** Already handled in `vector_indexer.py` with `batch_size=4000`. If you see it again, verify the batching loop is present.

### `TypeError: only 0-dimensional arrays can be converted to Python scalars`

**Cause:** NumPy ≥ 1.25 rejects implicit float conversion of 1×1 arrays.

**Fix:** In `app/bandit/linucb.py`, wrap matrix products in `.item()`:

```python
exploit = float((theta.T @ x).item())
```

### Test is "deselected" with no reason

**Cause:** LangSmith pytest plugin filtering.

**Fix:** Run with `-p no:langsmith`.

### PDF text is garbled or two-column

**Cause:** `PyPDFLoader` doesn't handle complex layouts.

**Fix:** Swap to `PyMuPDFLoader` (already the default in this project) or `PDFPlumberLoader`:

```python
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("path/to/dict.pdf")
```

### Ingestion returns 0 new articles on second run

**This is correct behavior** — SHA-256 deduplication is working.

### Hallucinated summaries

**Cause:** Retrieved context is too weak, LLM fills gaps from training data.

**Fix (planned):** Add similarity threshold in `app/rag/nodes.py`:

```python
docs_with_scores = _get_news_store().similarity_search_with_score(query, k=5)
relevant = [(d, s) for d, s in docs_with_scores if s < 1.2]
```

Also strengthen the prompt with "Do NOT add facts from your general knowledge."

---

##  Roadmap

### Phase 1 — Quality Guardrails (Week 1)

- [ ] Similarity threshold in retrieval nodes
- [ ] Strict "no hallucination" prompt
- [ ] "No relevant content" fallback path

### Phase 2 — Data Expansion (Week 2)

- [ ] Load remaining 7 dictionaries
- [ ] Fix/expand RSS feed list
- [ ] Add GNews API integration
- [ ] Fetch full article bodies (not just RSS summaries)

### Phase 3 — Fine-Tuning (Week 3–4)

- [ ] Extract term/definition pairs from dictionaries
- [ ] Build instruction dataset
- [ ] QLoRA fine-tune on Llama 3.1-8B or Qwen 2.5-7B
- [ ] Evaluate against base model

### Phase 4 — InShorts-Style Summarization (Week 5)

- [ ] Download InShorts dataset
- [ ] Build summarization pairs from corpus
- [ ] Fine-tune T5/Pegasus for 60-word summaries
- [ ] Integrate into RAG pipeline

### Phase 5 — Personalization (Week 6)

- [ ] Wire LinUCB to Streamlit feedback buttons
- [ ] Persist bandit state to disk
- [ ] Multi-user session tracking

### Phase 6 — Scale (Week 7+)

- [ ] RSSHub for sites without feeds
- [ ] Story clustering (semantic dedup)
- [ ] Redis/Kafka for producer/worker separation
- [ ] Local LLM via Ollama for bulk summarization

---

## Technology Stack

| Language      | Python 3.10+ |
| LLM           | Groq (`openai/gpt-oss-120b`) |
| Embeddings    | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Store  | ChromaDB (persistent, HNSW index) |
| Orchestration | LangGraph |
| PDF Loading   | PyMuPDFLoader |
| RSS Parsing   | feedparser |
| RL            | Custom LinUCB (NumPy) |
| UI            | Streamlit |
| Scheduling    | APScheduler |
| Config        | python-dotenv |

---

## Performance Notes

| Ingestion (8 feeds)         | 10–30 s | Network-bound |
| News embedding (850 chunks) | ~2 min | CPU, cached after first run |
| Dictionary embedding (14,697 chunks) | ~20 min | **Cached** to `.npy` — subsequent runs instant |
| ChromaDB query | 20–50 ms | HNSW index |
| LLM summary (Groq) | 3–15 s | Depends on output length |
| Full RAG pipeline | 5–20 s | End-to-end |

---

## Contributing

This is a personal research project. If you fork it:

1. Keep the modular structure — each phase is independently testable
2. Add sources to `app/config/settings.py`, not hardcoded
3. Cache expensive operations (embeddings, LLM responses)
4. Always verify with the Explore tab before trusting a summary

---

## ⚖️ Legal & Ethical Notes

- **RSS feeds** are generally safe to consume for personal use
- **APIs** (GNews, DIGITIMES, NewsAPI) have terms of service — respect rate limits
- **Bulk scraping** of paywalled content is **not** authorized by this project
- **Dictionary PDFs** — ensure you have the right to use them; copyright applies
- **Groq API** — free tier has rate limits; upgrade for production use

---

## 📄 License

MIT (or your preferred license) — add a `LICENSE` file to declare.

---

## 🙏 Acknowledgments

- [Groq](https://groq.com) for fast LLM inference
- [Sentence-Transformers](https://www.sbert.net) for the embedding model
- [ChromaDB](https://www.trychroma.com) for the vector store
- [LangChain](https://langchain.com) and [LangGraph](https://langchain-ai.github.io/langgraph/) for orchestration
- [Streamlit](https://streamlit.io) for the demo UI

---

**Built with** ⚡ **by an Electronics/Communication Engineer, for Electronics/Communication Engineers.**

<!-- # Electronics Technology Intelligence System

An AI-powered pipeline that:
1. Ingests electronics/technology news from RSS feeds
2. Fine-tunes an LLM on domain-specific dictionaries (QLoRA)
3. Generates technical summaries via RAG (LangGraph + ChromaDB)
4. Personalizes news selection with a LinUCB contextual bandit

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env
```

## Add Your Dictionaries

Place all 12 dictionary PDFs inside `dataset/`.

## Run

```bash
# 1. Ingest news
python scripts/run_ingestion.py

# 2. Index news + dictionaries
python scripts/run_indexing.py

# 3. Build fine-tuning dataset from dictionaries
python -c "from app.finetuning.dataset_builder import build; build()"

# 4. Fine-tune (requires GPU)
python scripts/run_finetune.py

# 5. Test the RAG pipeline
python scripts/run_rag_demo.py

# 6. Run daily scheduler
python scripts/run_scheduler.py
``` -->
