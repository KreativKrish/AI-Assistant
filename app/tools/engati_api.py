from langchain.tools import tool
from app.rag.retriever import get_retriever

@tool
def search_engati_docs(query: str) -> str:
    """
    Search the Engati API documentation and help articles for relevant information.
    Use this tool to find details about Engati APIs, features, configurations, or any
    platform-specific questions before answering the user.

    Args:
        query: The search query string.

    Returns:
        A string of the most relevant document passages found.
    """
    retriever = get_retriever()
    results = retriever.invoke(query)
    
    if not results:
        return "No relevant documentation found for the given query."
    
    # Format the results into a readable string
    formatted = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown Source")
        formatted.append(f"[Passage {i}] (Source: {source})\n{doc.page_content.strip()}")
    
    return "\n\n---\n\n".join(formatted)


def get_tools():
    """Returns the list of tools available to the agent."""
    return [search_engati_docs]

