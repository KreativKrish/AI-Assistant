import os
import asyncio
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from app.rag.retriever import get_retriever
from app.llm.prompt import get_prompt
from app.rag.ingest import ingest_docs

# Load environment variables from .env file
load_dotenv()

# --- App Setup ---
app = FastAPI(
    title="Kekda AI Assistant",
    description="An AI assistant that answers questions about any platform and APIs provided in context.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the chat frontend
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
DEFAULT_MODEL = "deepseek-r1-distill-llama-70b"

# Text models to exclude from selection (audio/image/guard models)
EXCLUDE_PATTERNS = ["whisper", "tts", "guard", "distil-whisper"]

def _infer_tier(model_id: str, ctx: int) -> tuple[str, str]:
    """Derive a (tier_key, tier_label) from the model id and context window."""
    mid = model_id.lower()
    if any(kw in mid for kw in ["deepseek-r1", "qwq", "qwen-qwq", "r1-"]):
        return "reasoning", "Reasoning Models"
    if ctx >= 100_000:
        return "large", "Large Context (100K+)"
    if ctx >= 30_000:
        return "mid", "Mid Context (30K–100K)"
    return "small", "Fast / Small Context (≤30K)"



# --- Request / Response Models ---
class ChatRequest(BaseModel):
    query: str
    model: str = DEFAULT_MODEL

class ChatResponse(BaseModel):
    answer: str
    source_documents: list[str] = []

class IngestRequest(BaseModel):
    url: str

# --- Per-model RAG chain cache ---
_chain_cache: dict = {}

def get_rag_chain(model_id: str = DEFAULT_MODEL):
    global _chain_cache
    if model_id not in _chain_cache:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Please add your Groq API key to the .env file."
            )

        llm = ChatGroq(
            model=model_id,
            temperature=0.9,
            groq_api_key=api_key,
        )
        prompt = get_prompt()
        retriever = get_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        _chain_cache[model_id] = create_retrieval_chain(retriever, document_chain)

    return _chain_cache[model_id]


# --- API Endpoints ---
@app.get("/", tags=["UI"])
async def serve_ui():
    """Serve the chat frontend UI."""
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "Kekda AI Assistant is running."}


@app.get("/api/models", tags=["Models"])
async def list_models():
    """Fetch models live from Groq API and group them by tier (derived from context window and model name)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set.")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                GROQ_MODELS_URL,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch models from Groq: {e}")

    tiers: dict = {}
    tier_order = ["reasoning", "large", "mid", "small"]

    for m in data.get("data", []):
        model_id: str = m.get("id", "")
        ctx: int = m.get("context_window", 0)
        active: bool = m.get("active", True)

        # Skip inactive models and non-text models (audio, vision-guard, etc.)
        if not active:
            continue
        # Filter out non-chat models by name pattern OR tiny context window (audio models have ctx ~448)
        if any(p in model_id.lower() for p in EXCLUDE_PATTERNS) or ctx < 1000:
            continue

        tier_key, tier_label = _infer_tier(model_id, ctx)

        if tier_key not in tiers:
            tiers[tier_key] = {"label": tier_label, "models": []}

        tiers[tier_key]["models"].append({
            "id": model_id,
            "ctx_window": ctx,
            "owned_by": m.get("owned_by", ""),
        })

    return {
        "default": DEFAULT_MODEL,
        "tiers": [
            {"key": k, "label": tiers[k]["label"], "models": tiers[k]["models"]}
            for k in tier_order if k in tiers
        ]
    }


@app.post("/api/ingest", tags=["Ingest"])
async def ingest(request: IngestRequest):
    """
    Ingest a new knowledge base from a URL.
    This wipes the existing database and crawls the new URL up to depth 3.
    """
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    try:
        chunks = await asyncio.to_thread(ingest_docs, request.url, True)

        # Bust chain cache so all models reload the new retriever
        global _chain_cache
        _chain_cache = {}

        return {"status": "success", "message": f"Successfully ingested {chunks} chunks from {request.url}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a question to the Kekda AI Assistant and receive a grounded answer.
    The assistant searches its knowledge base of provided context before responding.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        chain = get_rag_chain(request.model)
        result = chain.invoke({"input": request.query})

        sources = list({
            doc.metadata.get("source", "")
            for doc in result.get("context", [])
            if doc.metadata.get("source")
        })

        return ChatResponse(
            answer=result.get("answer", "I could not find an answer. Please try rephrasing."),
            source_documents=sources
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
