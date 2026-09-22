from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the sentence-transformers embedding model.
    First call downloads ~90MB from HuggingFace and caches it locally.
    Subsequent calls load from cache instantly.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # 90MB, 384-dim vectors, fast on CPU
        model_kwargs={"device": "cpu"},  # explicitly run on CPU
        encode_kwargs={"normalize_embeddings": True},  # L2-normalize vectors
        # normalization makes cosine similarity == dot product, slightly faster search
    )
    return embeddings


if __name__ == "__main__":
    print("Loading embedding model (first run downloads ~90MB)...")
    embeddings = get_embedding_model()

    # Test: embed two similar sentences and one unrelated one
    test_sentences = [
        "Prince is a software developer",
        "Prince works as a programmer",   # similar meaning → should be close
        "The weather is nice today",       # unrelated → should be far
    ]

    vectors = embeddings.embed_documents(test_sentences)

    print(f"\nEach sentence becomes a vector of {len(vectors[0])} numbers")
    print(f"First 5 numbers of sentence 1: {[round(v, 4) for v in vectors[0][:5]]}")
    print(f"First 5 numbers of sentence 2: {[round(v, 4) for v in vectors[1][:5]]}")
    print(f"First 5 numbers of sentence 3: {[round(v, 4) for v in vectors[2][:5]]}")

    # Compute cosine similarity manually to prove the concept
    import numpy as np
    v1, v2, v3 = [np.array(v) for v in vectors]
    sim_12 = float(np.dot(v1, v2))  # similar sentences
    sim_13 = float(np.dot(v1, v3))  # unrelated sentences

    print(f"\nSimilarity score (developer vs programmer) : {sim_12:.4f}  ← should be HIGH")
    print(f"Similarity score (developer vs weather)    : {sim_13:.4f}  ← should be LOW")
