from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Loading FAISS database...")

vector_store = FAISS.load_local(
    str(VECTORSTORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)

print("Connecting to Llama 3.2...")

llm = ChatOllama(
    model="llama3.2",
    temperature=0.2,
)

question = "Explain Ancient Egyptian architecture."

print("\nSearching knowledge base...")

docs = vector_store.similarity_search(
    question,
    k=3,
)

context = "\n\n".join(doc.page_content for doc in docs)

prompt = f"""
You are an architecture expert.

Answer ONLY using the provided context.

Context:

{context}

Question:

{question}

Provide a clear explanation suitable for a university student.
"""

print("\nGenerating answer...\n")

response = llm.invoke(prompt)

print("=" * 60)
print(response.content)
print("=" * 60)