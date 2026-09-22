from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: list) -> list:
    """
    Split a list of LangChain Documents into smaller chunks.
    Each chunk is still a Document object — same structure, just shorter page_content.
    Metadata (source file, page number) is automatically copied to every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # max characters per chunk
        chunk_overlap=100,    # characters shared between consecutive chunks
        length_function=len,  # how to measure size (character count, not tokens)
    )

    chunks = splitter.split_documents(documents)

    print(f"Documents before chunking : {len(documents)}")
    print(f"Chunks after chunking     : {len(chunks)}")
    return chunks


if __name__ == "__main__":
    import os
    from loader import load_documents

    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    docs = load_documents(data_path)
    chunks = chunk_documents(docs)

    # Inspect a few chunks to see what the splitter produced
    print("\n--- Sample chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}")
        print(f"Source   : {chunk.metadata.get('source', 'unknown')}")
        print(f"Length   : {len(chunk.page_content)} chars")
        print(f"Content  : {chunk.page_content[:200]}")
        print("-" * 40)
