"""Chat interface using FastAPI + Ollama RAG.

Endpoints:
- POST /api/chat — RAG chat against email database
- GET /api/stats — Qdrant collection stats
- GET /api/pipeline/status — full pipeline state
- GET /api/pipeline/progress — stage counts + percentages
- GET /api/pipeline/errors — failed emails
- POST /api/pipeline/resume — trigger resume run (background)
- POST /api/pipeline/pause — pause the pipeline
"""

import threading

import ollama
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from config.settings import config
from src.embedding.embedder import EmailEmbedder
from src.storage.vector_store import EmailVectorStore
from src.tracking.state import PipelineStateManager, PipelineStatus

app = FastAPI(title="Chat with Your Emails")

embedder = EmailEmbedder()
store = EmailVectorStore()
llm_client = ollama.Client(host=config.ollama.base_url)

SYSTEM_PROMPT = """You are a helpful email assistant. You have access to the user's email database.
Answer questions based on the provided email context. Be concise and accurate.
If the context doesn't contain relevant information, say so clearly.

When referencing emails, mention the sender, date, and subject when available."""


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str
    sources: list[dict] = []


# ── Chat endpoints ──

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    query_embedding = embedder.embed_text(request.message)
    results = store.search(query_embedding, limit=5)

    context_parts = []
    sources = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[Email {i+1}] From: {r['sender']} | Date: {r['date']} | Subject: {r['subject']}\n"
            f"Category: {r['category']}\n"
            f"{r['text']}"
        )
        sources.append({
            "subject": r["subject"],
            "sender": r["sender"],
            "date": r["date"],
            "score": round(r["score"], 3),
        })

    context = "\n\n---\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.history[-10:]:
        messages.append(msg)
    messages.append({
        "role": "user",
        "content": f"Context from emails:\n\n{context}\n\nQuestion: {request.message}",
    })

    response = llm_client.chat(
        model=config.models.chat,
        messages=messages,
        options={"temperature": 0.3},
    )

    return ChatResponse(
        response=response["message"]["content"],
        sources=sources,
    )


@app.get("/api/stats")
async def stats():
    return store.get_collection_info()


# ── Pipeline status endpoints ──

@app.get("/api/pipeline/status")
async def pipeline_status():
    """Full pipeline state."""
    state = PipelineStateManager()
    return state.get_progress()


@app.get("/api/pipeline/progress")
async def pipeline_progress():
    """Stage counts and percentages."""
    state = PipelineStateManager()
    progress = state.get_progress()
    return {
        "status": progress["status"],
        "total": progress["total_emails"],
        "overall_pct": progress["overall_pct"],
        "stages": progress["stages"],
        "errors": progress["total_errors"],
    }


@app.get("/api/pipeline/errors")
async def pipeline_errors():
    """List all failed emails with errors."""
    state = PipelineStateManager()
    return {"errors": state.get_failed_emails()}


@app.post("/api/pipeline/resume")
async def pipeline_resume(background_tasks: BackgroundTasks):
    """Trigger a pipeline resume in the background."""
    state = PipelineStateManager()
    if state.status == PipelineStatus.RUNNING.value:
        return {"status": "already_running", "message": "Pipeline is already running"}

    state.set_status(PipelineStatus.RUNNING)

    def _run_resume():
        import json
        from datetime import datetime
        from src.preprocessing.pipeline import PreprocessingPipeline

        state = PipelineStateManager()
        raw_dir = "data/raw_emails"
        emails = []
        for filename in sorted(__import__("os").listdir(raw_dir)):
            if filename.endswith(".json"):
                with open(f"{raw_dir}/{filename}") as f:
                    email = json.load(f)
                    email["date"] = datetime.fromisoformat(email["date"])
                    emails.append(email)

        pipeline = PreprocessingPipeline(state_manager=state)
        pipeline.run(emails)

    background_tasks.add_task(_run_resume)
    return {"status": "resumed", "message": "Pipeline resume started in background"}


@app.post("/api/pipeline/pause")
async def pipeline_pause():
    """Pause the pipeline (checked on next email)."""
    state = PipelineStateManager()
    state.set_status(PipelineStatus.PAUSED)
    return {"status": "paused", "message": "Pipeline will pause after current email completes"}


# ── Server startup ──

def main():
    """Start the chat server."""
    import uvicorn
    from rich.console import Console
    console = Console()

    info = store.get_collection_info()
    console.print(f"[bold]Chat with Your Emails[/bold]")
    console.print(f"Vector store: {info['points_count']} chunks indexed")
    console.print(f"Chat model: {config.models.chat}")
    console.print(f"Starting server on http://0.0.0.0:8501")

    uvicorn.run(app, host="0.0.0.0", port=8501)


if __name__ == "__main__":
    main()
