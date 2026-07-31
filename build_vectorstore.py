from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_text_documents() -> list[Document]:
    """Read all .txt files from the knowledge_base folder."""

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge-base folder not found:\n{KNOWLEDGE_BASE_DIR}"
        )

    text_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(
            "No .txt files were found inside the knowledge_base folder."
        )

    documents: list[Document] = []

    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue

        document = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "style": file_path.stem.replace("_", " ").title(),
            },
        )

        documents.append(document)
        print(f"Loaded: {file_path.name}")

    if not documents:
        raise ValueError("All knowledge-base files were empty.")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split long documents into smaller overlapping chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    """Load the local Sentence Transformers embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def main() -> None:
    print("\nBuilding architectural knowledge vector store")
    print("---------------------------------------------")

    documents = load_text_documents()

    print(f"\nLoaded {len(documents)} knowledge-base documents.")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} text chunks.")

    print("\nLoading embedding model...")
    print("The first run may download the embedding model.")

    embeddings = create_embeddings()

    print("Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(VECTORSTORE_DIR))

    print("\n✅ Vector store created successfully!")
    print(f"Saved to: {VECTORSTORE_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Indexed chunks: {len(chunks)}")

    print("\nGenerated files:")

    for file_path in VECTORSTORE_DIR.iterdir():
        print(f"- {file_path.name}")


if __name__ == "__main__":
    main()