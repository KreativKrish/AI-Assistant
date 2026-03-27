import os
import re
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db")

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return re.sub(r"\n\n+", "\n\n", soup.get_text()).strip()

def ingest_docs(base_url="https://help.engati.ai/", clear=False):
    print(f"Recursively loading content from {base_url}...")
    
    if clear and os.path.exists(PERSIST_DIRECTORY):
        print("Clearing existing knowledge base...")
        shutil.rmtree(PERSIST_DIRECTORY)
        print("Cleared.")

    loader = RecursiveUrlLoader(
        url=base_url,
        max_depth=3, # Configures how deep the crawler should go from the starting URL
        extractor=bs4_extractor,
    )
    docs = loader.load()
    if not docs:
        print("No documents found at this URL.")
        return 0

    print(f"Loaded {len(docs)} page(s). Splitting text...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks. Generating embeddings (local model)...")

    # Use a local HuggingFace embedding model — no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    print(f"✅ Vector store persisted to {PERSIST_DIRECTORY}")
    print(f"   Total chunks stored: {len(splits)}")
    return len(splits)


if __name__ == "__main__":
    ingest_docs()
