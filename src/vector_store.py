import os
from langchain_chroma import Chroma

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def build_vector_store(chunks: list, embedding_model) -> Chroma:
    """
    Embed all chunks and save them to ChromaDB on disk.
    Call this ONCE to build the store for the first time.
    """
    print(f"Embedding {len(chunks)} chunks and saving to disk...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store saved to: {CHROMA_DIR}")
    return vector_store


def load_vector_store(embedding_model) -> Chroma:
    """
    Load an already-built ChromaDB from disk.
    No re-embedding happens — just reads the saved vectors.
    """
    print("Loading existing vector store from disk...")
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )
    print(f"Loaded. Collection has {vector_store._collection.count()} chunks.")
    return vector_store


def get_or_build_vector_store(chunks: list, embedding_model) -> Chroma:
    """
    Smart loader: build on first run, reload on every run after that.
    This is what the main chat app will call.
    """
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return load_vector_store(embedding_model)
    else:
        return build_vector_store(chunks, embedding_model)


if __name__ == "__main__":
    import shutil
    from loader import load_documents
    from chunker import chunk_documents
    from embedder import get_embedding_model

    data_path = os.path.join(os.path.dirname(__file__), "..", "data")

    # Fresh build for testing — wipe any existing store first
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("Cleared old chroma_db/ for a clean test.")

    docs = load_documents(data_path)
    chunks = chunk_documents(docs)
    embeddings = get_embedding_model()

    store = build_vector_store(chunks, embeddings)

    # Prove it works: run a raw similarity search
    print("\n--- Test similarity search ---")
    query = "What does Prince do professionally?"
    results = store.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1} (from: {doc.metadata.get('source', '?').split(chr(92))[-1]})")
        print(doc.page_content[:250])
