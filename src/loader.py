import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader


def load_documents(data_dir: str) -> list:
    """
    Walk through the data/ folder, pick the right loader for each file type,
    and return a flat list of LangChain Document objects.
    """
    documents = []

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".txt"):
            # TextLoader reads the whole file as one Document
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded TXT: {filename} → {len(docs)} document(s)")

        elif filename.endswith(".pdf"):
            # PyPDFLoader creates one Document per page
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded PDF: {filename} → {len(docs)} document(s)")

        else:
            print(f"Skipped (unsupported type): {filename}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    # This block only runs when you execute this file directly.
    # It's our isolated test for this step.
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    docs = load_documents(data_path)

    # Inspect the first document in detail
    if docs:
        print("\n--- Inspecting first Document ---")
        print(f"Metadata : {docs[0].metadata}")
        print(f"Content preview : {docs[0].page_content[:300]}")
