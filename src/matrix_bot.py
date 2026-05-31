"""Matrix bot for chatting with your emails.

Connects to Matrix Synapse, listens for messages, queries email RAG, replies.

Usage: email-matrix
"""

import asyncio
import hashlib
import hmac
import json
import os
import sys
import time

import requests
from nio import AsyncClient, MatrixRoom, RoomMessageText, LoginResponse, MatrixInvitedRoom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config
from src.embedding.embedder import EmailEmbedder
from src.storage.vector_store import EmailVectorStore
import ollama

HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "http://192.168.0.250:8008")
USER_ID = os.environ.get("MATRIX_USER_ID", "@bot:7anand7s.com")
USER = os.environ.get("MATRIX_USER", "bot")
PASSWORD = os.environ.get("MATRIX_PASSWORD", "bot123")
ADMIN_USER = os.environ.get("MATRIX_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("MATRIX_ADMIN_PASSWORD", "admin123")
STORE_PATH = "data/matrix_store"

SYSTEM_PROMPT = """You are a helpful email assistant. Answer based on the email context provided.
Be concise. Reference sender, date, and subject. If context is insufficient, say so.
Format for chat — use bullet points, bold for key info."""

embedder = EmailEmbedder()
store = EmailVectorStore()
llm_client = ollama.Client(host=config.ollama.base_url)


def get_admin_token() -> str:
    """Get admin access token via registration shared secret."""
    r = requests.post(f"{HOMESERVER}/_matrix/client/r0/login", json={
        "type": "m.login.password",
        "identifier": {"type": "m.id.user", "user": ADMIN_USER},
        "password": ADMIN_PASSWORD,
    })
    if r.status_code == 200:
        return r.json()["access_token"]
    raise RuntimeError(f"Admin login failed: {r.json()}")


def get_bot_token_via_admin() -> str:
    """Get a bot access token via admin API (bypasses login rate limits)."""
    admin_token = get_admin_token()
    r = requests.post(
        f"{HOMESERVER}/_synapse/admin/v1/users/@{USER}:{HOMESERVER.split('//')[1].split(':')[0]}/login",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={},
    )
    if r.status_code == 200:
        return r.json()["access_token"]
    raise RuntimeError(f"Bot token creation failed: {r.json()}")


def query_emails(question: str) -> str:
    """Query email RAG pipeline."""
    try:
        info = store.get_collection_info()
        if info["points_count"] == 0:
            return "No emails embedded yet. Run `email-embed` first."

        query_embedding = embedder.embed_text(question)
        results = store.search(query_embedding, limit=5)

        if not results:
            return "No relevant emails found."

        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(
                f"[Email {i+1}] From: {r['sender']} | Date: {r['date']} | Subject: {r['subject']}\n"
                f"Category: {r['category']}\n{r['text'][:2000]}"
            )
        context = "\n\n---\n\n".join(context_parts)

        response = llm_client.chat(
            model=config.models.chat,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Email context:\n\n{context}\n\nQuestion: {question}"},
            ],
            options={"temperature": 0.3},
        )

        answer = response["message"]["content"]
        sources = [f"- {r['subject'][:50]} — {r['sender'][:30]}" for r in results]
        return f"{answer}\n\nSources:\n" + "\n".join(sources)

    except Exception as e:
        return f"Error: {e}"


class EmailBot:
    def __init__(self):
        os.makedirs(STORE_PATH, exist_ok=True)
        self.client = AsyncClient(HOMESERVER, USER_ID, store_path=STORE_PATH)
        self.room_id = None

    async def login(self):
        """Log in using admin API to bypass rate limits."""
        # Try restoring session first
        try:
            await self.client.restore_login()
            print(f"Restored session for {self.client.user_id}")
            return
        except Exception:
            pass

        # Get token via admin API (bypasses login rate limits)
        try:
            token = get_bot_token_via_admin()
            self.client.access_token = token
            self.client.user_id = USER_ID
            print(f"Got bot token via admin API")
            return
        except Exception as e:
            print(f"Admin API failed: {e}")

        # Fallback to direct login
        response = await self.client.login(PASSWORD, device_name="email-bot")
        if isinstance(response, LoginResponse):
            print(f"Logged in as {response.user_id}")
        else:
            print(f"Login failed: {response}")
            sys.exit(1)

    async def on_invite(self, room: MatrixInvitedRoom):
        """Auto-join when invited."""
        print(f"Invited to room {room.room_id}, joining...")
        resp = await self.client.join(room.room_id)
        print(f"Joined: {resp}")
        self.room_id = room.room_id
        await self.client.room_send(
            room_id=room.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": "Hi! I'm your email assistant. Ask me anything about your emails."},
        )

    async def on_message(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming messages."""
        if event.sender == USER_ID:
            return

        question = event.body.strip()
        if not question:
            return

        print(f"[{room.display_name}] {event.sender}: {question}")

        # Query emails (non-blocking)
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, query_emails, question)

        # Send reply
        await self.client.room_send(
            room_id=room.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": answer},
        )
        print(f"[Bot]: {answer[:100]}...")

    async def run(self):
        """Start the bot."""
        await self.login()

        self.client.add_event_callback(self.on_invite, MatrixInvitedRoom)
        self.client.add_event_callback(self.on_message, RoomMessageText)

        print("Syncing...")
        await self.client.sync(timeout=30000, full_state=True)

        for room_id, room in self.client.rooms.items():
            print(f"In room: {room.display_name} ({room_id})")

        print("Listening for messages...")
        await self.client.sync_forever(timeout=30000)


async def _run():
    bot = EmailBot()
    await bot.run()


def main():
    """CLI: email-matrix"""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
