"""Chat interface using FastAPI + Ollama RAG."""

import ollama
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import config
from src.embedding.embedder import EmailEmbedder
from src.storage.vector_store import EmailVectorStore

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
    history: list[dict] = []  # [{"role": "user"/"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    response: str
    sources: list[dict] = []


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 1. Embed the query
    query_embedding = embedder.embed_text(request.message)

    # 2. Search for relevant email chunks
    results = store.search(query_embedding, limit=5)

    # 3. Build context from search results
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

    # 4. Generate response with LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    for msg in request.history[-10:]:  # Keep last 10 messages for context
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
