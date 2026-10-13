from app.pipeline.daily_pipeline import run_indexing
from app.ingestion.dictionary_loader import load_dictionaries, chunk_documents
from app.ingestion.vector_indexer import index_dictionary_chunks
from app.config.settings import PATHS

if __name__ == "__main__":
    # Index news
    run_indexing()

    # Index dictionaries
    print(f"Loading dictionary PDFs from {PATHS['dataset']}")
    pages = load_dictionaries(PATHS["dataset"])
    if pages:
        chunks = chunk_documents(pages)
        index_dictionary_chunks(chunks)

## This is the main entry point for indexing news articles and dictionary PDFs. It first runs the `run_indexing()` function to index news articles, and then it loads dictionary PDFs from the specified dataset path, chunks them into smaller pieces, and indexes those chunks using the `index_dictionary_chunks()` function.