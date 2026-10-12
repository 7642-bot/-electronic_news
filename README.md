# Electronics Technology Intelligence System

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
```
