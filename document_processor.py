from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import all constants from our single config file
import config

# --- INDEXING FUNCTIONS ---

def find_files(path: Path) -> list[Path]:
    """Finds all supported documents in a given path."""
    if path.is_file():
        return [path]
    exts = [".txt", ".md", ".pdf"]
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def load_documents(paths: list[Path]) -> list[Document]:
    """Loads documents from the given paths."""
    print(f"Loading {len(paths)} document(s)...")
    docs: list[Document] = []
    for p in paths:
        try:
            if p.suffix.lower() in [".txt", ".md"]:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return docs

def split_documents(docs: list[Document]) -> list[Document]:
    """Splits the documents into smaller chunks using config settings."""
    print(f"Splitting {len(docs)} document(s) into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)

def build_and_save_faiss(chunks: list[Document]):
    """Builds and saves the FAISS vector store using Hugging Face embeddings."""
    print("Initializing Hugging Face embeddings model...")
    # --- MODIFIED: Use HuggingFaceEmbeddings ---
    # The model will be downloaded automatically the first time you run this.
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    
    print(f"Building FAISS index from {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    config.INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(config.INDEX_PATH))
    print(f"✅ Saved FAISS index to: {config.INDEX_PATH.resolve()}")
    return vector_store

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("--- Starting Indexing Process ---")
    
    # 1. Find and load the document(s)
    doc_paths = find_files(config.DOCS_PATH)
    documents = load_documents(doc_paths)
    
    # 2. Split the documents into chunks
    chunks = split_documents(documents)
    
    # 3. Build the FAISS index and save it
    if chunks:
        build_and_save_faiss(chunks)
    else:
        print("No chunks to index. Please check your documents.")
        
    print("--- Indexing Process Complete ---")