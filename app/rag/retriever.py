import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db")


def get_retriever():
    """
    Loads the persisted Chroma vector store and returns a retriever interface.
    Uses local HuggingFace embeddings — no API key required.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )

    # Return top 4 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 4})
