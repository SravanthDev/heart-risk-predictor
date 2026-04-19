"""
build_vectordb.py
=================
Dynamically scans data/ for ALL .pdf files, chunks them, embeds them with
sentence-transformers (directly — no torch DataLoader forking), and saves a
FAISS index to data/faiss_index/.

Run ONCE locally, then commit data/faiss_index/ to Git so Streamlit Cloud
can load the index at runtime without rebuilding.

Usage
-----
    cd heart-risk-predictor/          # repo root
    python data/build_vectordb.py

Optional flags
--------------
    --chunk-size    INT   characters per chunk             (default: 800)
    --chunk-overlap INT   character overlap between chunks (default: 100)
    --top-k         INT   test query top-k                 (default: 3)
    --model         STR   HuggingFace embedding model      (default: sentence-transformers/all-MiniLM-L6-v2)
    --no-test             skip the smoke-test query

Example
-------
    python data/build_vectordb.py --chunk-size 600 --chunk-overlap 80
"""

# ── Prevent macOS semaphore segfault from torch / tokenizer multiprocessing ───
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import sys
import json
import pickle
import numpy as np
from pathlib import Path

# ── Path constants ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()   # data/
REPO_ROOT  = SCRIPT_DIR.parent                 # heart-risk-predictor/
INDEX_PATH = SCRIPT_DIR / "faiss_index"


# ── CLI args ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS vector index from PDFs in data/")
    parser.add_argument("--chunk-size",    type=int, default=800,
                        help="Characters per text chunk (default: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=100,
                        help="Overlap between consecutive chunks (default: 100)")
    parser.add_argument("--top-k",         type=int, default=3,
                        help="Top-k results for smoke-test query (default: 3)")
    parser.add_argument("--model",         type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformers model name")
    parser.add_argument("--no-test",       action="store_true",
                        help="Skip smoke-test retrieval query")
    return parser.parse_args()


# ── Import check ───────────────────────────────────────────────────────────────
def check_imports():
    missing = []
    for pkg, install_name in [
        ("pypdf",                "pypdf"),
        ("sentence_transformers","sentence-transformers"),
        ("faiss",                "faiss-cpu"),
        ("langchain_community",  "langchain-community"),
        ("langchain",            "langchain"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(install_name)
    if missing:
        print(f"[ERROR] Missing packages: {missing}")
        print("Run:  python3 -m pip install -r requirements.txt")
        sys.exit(1)

check_imports()

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


# ── Dynamic PDF discovery ──────────────────────────────────────────────────────
def discover_pdfs(data_dir: Path) -> list[Path]:
    """Find all .pdf files in data_dir (skips faiss_index/ subdir)."""
    return [
        p for p in sorted(data_dir.rglob("*.pdf"))
        if "faiss_index" not in p.parts
    ]


# ── Load PDFs with pypdf directly (no LangChain loader) ───────────────────────
def load_pdfs(pdf_paths: list[Path]) -> list[Document]:
    all_docs = []
    for pdf_path in pdf_paths:
        size_kb = pdf_path.stat().st_size / 1024
        print(f"  📄 Loading: {pdf_path.name} ({size_kb:.1f} KB)")
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    all_docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num + 1,
                            "file_path": str(pdf_path),
                        }
                    ))
            print(f"          → {len(reader.pages)} pages loaded")
        except Exception as e:
            print(f"  [WARN]  Could not load {pdf_path.name}: {e}")
    return all_docs


# ── Split into chunks ──────────────────────────────────────────────────────────
def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


# ── Embed with SentenceTransformers directly (no torch fork) ──────────────────
def embed_chunks(chunks: list[Document], model_name: str) -> tuple[np.ndarray, list[Document]]:
    print(f"  🔢 Model      : {model_name}")
    print(f"     Chunks     : {len(chunks)}")
    print(f"     Loading model (first run downloads ~90 MB) ...")

    model = SentenceTransformer(model_name)
    texts = [chunk.page_content for chunk in chunks]

    print(f"     Encoding {len(texts)} chunks (single-threaded, macOS-safe) ...")
    # sentence-transformers v5 removed num_workers from encode().
    # Single-process encoding is safe on macOS (no fork/semaphore issues).
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32), chunks


# ── Build & save FAISS index ───────────────────────────────────────────────────
def build_and_save(embeddings: np.ndarray, chunks: list[Document]) -> LangchainFAISS:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # Inner-product (cosine with normalized vecs)
    index.add(embeddings)

    # Build the LangChain FAISS wrapper so app.py can load it the same way
    docstore = InMemoryDocstore({str(i): chunks[i] for i in range(len(chunks))})
    index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

    # We store embeddings using a dummy embedding function wrapper
    # (LangchainFAISS.load_local needs it at runtime, but we supply the index directly)
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_fn = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = LangchainFAISS(
        embedding_function=embedding_fn,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_PATH))
    print(f"  ✅ Index saved to: {INDEX_PATH}/")
    print(f"     Files: {[f.name for f in INDEX_PATH.iterdir()]}")
    return db


# ── Smoke test ─────────────────────────────────────────────────────────────────
def smoke_test(db: LangchainFAISS, top_k: int):
    queries = [
        "cardiovascular risk factors hypertension cholesterol",
        "heart disease prevention guidelines",
        "blood pressure management treatment",
    ]
    print("\n── Smoke Test ──────────────────────────────────────────────────")
    for query in queries:
        results = db.similarity_search(query, k=top_k)
        print(f"\n  Query: \"{query}\"")
        for i, doc in enumerate(results, 1):
            src  = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i}] {src} p.{page}: {preview}...")
    print("\n────────────────────────────────────────────────────────────────")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 64)
    print("  Heart Risk Predictor — FAISS Vector DB Builder")
    print("=" * 64)
    print(f"\n  Scanning : {SCRIPT_DIR}")
    print(f"  Output   : {INDEX_PATH}")
    print(f"  Chunk    : size={args.chunk_size}, overlap={args.chunk_overlap}")

    # 1. Discover PDFs
    pdf_paths = discover_pdfs(SCRIPT_DIR)
    if not pdf_paths:
        print("\n[ERROR] No PDF files found in data/.")
        sys.exit(1)

    print(f"\n  Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"    • {p.name}")

    # 2. Load
    print("\n── Loading PDFs ────────────────────────────────────────────────")
    docs = load_pdfs(pdf_paths)
    if not docs:
        print("[ERROR] All PDFs failed to load.")
        sys.exit(1)
    print(f"\n  Total pages loaded: {len(docs)}")

    # 3. Split
    print("\n── Splitting into chunks ───────────────────────────────────────")
    chunks = split_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"  Total chunks: {len(chunks)}")

    # 4. Embed (single-threaded, safe on macOS)
    print("\n── Embedding chunks ────────────────────────────────────────────")
    embeddings_arr, chunks = embed_chunks(chunks, args.model)

    # 5. Build & save
    print("\n── Building FAISS index ────────────────────────────────────────")
    db = build_and_save(embeddings_arr, chunks)

    # 6. Smoke test
    if not args.no_test:
        smoke_test(db, args.top_k)

    print("\n  🎉 Done! Commit data/faiss_index/ to Git for Streamlit Cloud.\n")


if __name__ == "__main__":
    main()
