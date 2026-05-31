"""Matrix bot for chatting with your emails.

Connects to your Matrix Synapse server, listens for messages in a room,
queries the email RAG pipeline, and replies back.

Usage:
    python3 -m src.matrix_bot

Environment variables (or config):
    MATRIX_HOMESERVER  - Synapse URL (default: http://192.168.0.250:8008)
    MATRIX_USER        - Bot username (default: 7anand7s)
    MATRIX_PASSWORD    - Bot password
    MATRIX_ROOM        - Room ID or name to listen in (auto-joins if invited)
"""

import asyncio
import os
import sys
from nio import AsyncClient, MatrixRoom, RoomMessageText, LoginResponse, InviteMemberEvent

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config
from src.embedding.embedder import EmailEmbedder
from src.storage.vector_store import EmailVectorStore
import ollama

# ── Configuration ──

HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "http://192.168.0.250:8008")
USER_ID = os.environ.get("MATRIX_USER_ID", "@bot:7anand7s.com")
USER = os.environ.get("MATRIX_USER", "bot")
PASSWORD = os.environ.get("MATRIX_PASSWORD", "bot123")
STORE_PATH = os.environ.get("MATRIX_STORE", "data/matrix_store")

SYSTEM_PROMPT = """You are a helpful email assistant. You have access to the user's email database.
Answer questions based on the provided email context. Be concise and accurate.
If the context doesn't contain relevant information, say so clearly.
When referencing emails, mention the sender, date, and subject when available.
Format your responses nicely for chat — use bullet points, bold for important info."""

# ── RAG Pipeline ──

embedder = EmailEmbedder()
store = EmailVectorStore()
llm_client = ollama.Client(host=config.ollama.base_url)


def query_emails(question: str) -> str:
    """Query the email RAG pipeline and return a response."""
    try:
        # Check if we have data
        info = store.get_collection_info()
        if info["points_count"] == 0:
            return "No emails have been embedded yet. Run `email-embed` first."

        # Retrieve relevant chunks
        query_embedding = embedder.embed_text(question)
        results = store.search(query_embedding, limit=5)

        if not results:
            return "I couldn't find any relevant emails for that question."

        # Build context
        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(
                f"[Email {i+1}] From: {r['sender']} | Date: {r['date']} | Subject: {r['subject']}\n"
                f"Category: {r['category']}\n"
                f"{r['text'][:2000]}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Generate response
        response = llm_client.chat(
            model=config.models.chat,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Email context:\n\n{context}\n\nQuestion: {question}"},
            ],
            options={"temperature": 0.3},
        )

        answer = response["message"]["content"]

        # Add sources
        sources = []
        for r in results:
            sources.append(f"• {r['subject'][:50]} — {r['sender'][:30]}")

        return f"{answer}\n\n**Sources:**\n" + "\n".join(sources)

    except Exception as e:
        return f"Error querying emails: {e}"


# ── Matrix Bot ──

class EmailBot:
    def __init__(self):
        self.client = AsyncClient(HOMESERVER, USER_ID)
        self.client.store_path = STORE_PATH
        self.room_id = None

    async def login(self):
        """Log in to Matrix."""
        os.makedirs(STORE_PATH, exist_ok=True)
        response = await self.client.login(PASSWORD)
        if isinstance(response, LoginResponse):
            print(f"Logged in as {USER}")
        else:
            print(f"Login failed: {response}")
            sys.exit(1)

    async def on_invite(self, event: InviteMemberEvent, room: MatrixRoom):
        """Auto-join rooms when invited."""
        if event.state_key == self.client.user_id:
            print(f"Joining room: {room.display_name} ({room.room_id})")
            await self.client.join(room.room_id)
            self.room_id = room.room_id
            await self.client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": "Hi! I'm your email assistant. Ask me anything about your emails.",
                },
            )

    async def on_message(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming messages."""
        # Ignore own messages
        if event.sender == self.client.user_id:
            return

        question = event.body.strip()
        if not question:
            return

        print(f"[{room.display_name}] {event.sender}: {question}")

        # Send typing indicator
        await self.client.room_send(
            room_id=room.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.typing", "body": ""},
        )

        # Query emails (run in executor to not block)
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, query_emails, question)

        # Send response
        await self.client.room_send(
            room_id=room.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": answer},
        )

        print(f"[Bot]: {answer[:100]}...")

    async def run(self):
        """Start the bot."""
        await self.login()

        # Set up event callbacks
        self.client.add_event_callback(self.on_invite, InviteMemberEvent)
        self.client.add_event_callback(self.on_message, RoomMessageText)

        # Initial sync
        print("Syncing...")
        await self.client.sync(timeout=10000)

        # Auto-join any rooms we've been invited to
        print(f"Listening for messages... (user: {USER})")
        print(f"Invite this user to a room to start chatting!")

        # Sync forever
        await self.client.sync_forever(timeout=30000)


async def _run():
    bot = EmailBot()
    await bot.run()


def main():
    """CLI entry point: email-matrix"""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
