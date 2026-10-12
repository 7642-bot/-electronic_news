import glob
import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_dictionaries(folder: Path) -> list:
    pdf_paths = sorted(glob.glob(os.path.join(str(folder), "*.pdf")))
    if not pdf_paths:
        print(f"⚠ No PDFs found in {folder}. Add your dictionaries here.")
        return []

    all_docs = []
    for path in pdf_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            loader = PyMuPDFLoader(path)
            pages = loader.load()
            for p in pages:
                p.metadata["dictionary_name"] = name
                p.metadata["type"] = "dictionary"
            all_docs.extend(pages)
            print(f"✓ {name}: {len(pages)} pages")
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
    return all_docs

def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
