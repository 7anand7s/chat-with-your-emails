"""Matrix bot for chatting with your emails.

Connects to Matrix Synapse, listens for messages, queries email RAG, replies.
Uses HTML formatting for clean display in Element.

Usage: email-matrix
"""

import asyncio
import json
import os
import sys
import re

import requests
from nio import AsyncClient, MatrixRoom, RoomMessageText, MatrixInvitedRoom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config
from src.embedding.embedder import EmailEmbedder
from src.storage.vector_store import EmailVectorStore
import ollama

HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "http://192.168.0.250:8008")
USER_ID = os.environ.get("MATRIX_USER_ID", "@bot:7anand7s.com")
ADMIN_USER = os.environ.get("MATRIX_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("MATRIX_ADMIN_PASSWORD", "admin123")
STORE_PATH = "data/matrix_store"

SYSTEM_PROMPT = """You are a helpful email assistant. Answer based on the email context provided.
Be concise. Reference sender, date, and subject.

FORMATTING RULES:
- Use - for bullet points (never use * or numbers)
- Use **text** for bold
- NEVER use angle brackets < or > — they break the chat display
- NEVER use markdown headers (# ## ###)
- NEVER use code blocks or backticks
- NEVER use HTML tags
- Keep it clean and readable as plain text with bullets and bold only"""

embedder = EmailEmbedder()
store = EmailVectorStore()
llm_client = ollama.Client(host=config.ollama.base_url)


def get_admin_token() -> str:
    """Get admin access token."""
    r = requests.post(f"{HOMESERVER}/_matrix/client/r0/login", json={
        "type": "m.login.password",
        "identifier": {"type": "m.id.user", "user": ADMIN_USER},
        "password": ADMIN_PASSWORD,
    })
    if r.status_code == 200:
        return r.json()["access_token"]
    raise RuntimeError(f"Admin login failed: {r.json()}")


def get_bot_token() -> str:
    """Get bot access token via admin API (bypasses login rate limits)."""
    admin_token = get_admin_token()
    r = requests.post(
        f"{HOMESERVER}/_synapse/admin/v1/users/{USER_ID}/login",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={},
    )
    if r.status_code == 200:
        return r.json()["access_token"]
    raise RuntimeError(f"Bot token failed: {r.json()}")


def markdown_to_html(text: str) -> str:
    """Convert simple markdown to Matrix HTML."""
    # Escape HTML special chars
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Bold: **text** → <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Bullet points: - text OR * text → <li>text</li>
    lines = text.split('\n')
    result = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        # Match both "- " and "* " (with optional extra spaces)
        bullet_match = re.match(r'^[-*]\s{1,4}(.+)', stripped)
        if bullet_match:
            if not in_list:
                result.append('<ul>')
                in_list = True
            result.append(f'<li>{bullet_match.group(1)}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            if stripped:
                result.append(f'<p>{stripped}</p>')
            else:
                result.append('<br/>')
    if in_list:
        result.append('</ul>')

    return ''.join(result)


def send_matrix_message(client, room_id: str, text: str):
    """Send a message with HTML formatting."""
    html = markdown_to_html(text)
    return client.room_send(
        room_id=room_id,
        message_type="m.room.message",
        content={
            "msgtype": "m.text",
            "body": text,
            "format": "org.matrix.custom.html",
            "formatted_body": html,
        },
    )


def query_emails(question: str) -> str:
    """Query email RAG pipeline."""
    try:
        info = store.get_collection_info()
        if info["points_count"] == 0:
            return "No emails embedded yet. Run `email-embed` first."

        query_embedding = embedder.embed_text(question)
        results = store.search(query_embedding, limit=5)

        if not results:
            return "No relevant emails found for that question."

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
        # Strip angle brackets from LLM output (breaks HTML rendering)
        answer = answer.replace("<", "(").replace(">", ")")
        sources = [f"- {r['subject']} — {r['sender']}" for r in results]
        return f"{answer}\n\n**Sources:**\n" + "\n".join(sources)

    except Exception as e:
        return f"Error: {e}"


class EmailBot:
    def __init__(self):
        os.makedirs(STORE_PATH, exist_ok=True)
        self.client = AsyncClient(HOMESERVER, USER_ID, store_path=STORE_PATH)

    async def login(self):
        """Log in using admin API (bypasses rate limits)."""
        try:
            await self.client.restore_login()
            print(f"Restored session for {self.client.user_id}")
            return
        except Exception:
            pass

        try:
            token = get_bot_token()
            self.client.access_token = token
            self.client.user_id = USER_ID
            print(f"Logged in via admin API")
            return
        except Exception as e:
            print(f"Login failed: {e}")
            sys.exit(1)

    async def on_invite(self, room: MatrixInvitedRoom):
        """Auto-join when invited."""
        print(f"Invited to {room.room_id}, joining...")
        await self.client.join(room.room_id)
        await send_matrix_message(
            self.client, room.room_id,
            "Hi! I'm your email assistant. Ask me anything about your emails."
        )

    async def on_message(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming messages."""
        # Ignore own messages
        if event.sender == USER_ID:
            return

        question = event.body.strip()
        if not question:
            return

        print(f"[{room.display_name}] {event.sender}: {question}")

        # Query RAG
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, query_emails, question)

        # Send formatted reply
        await send_matrix_message(self.client, room.room_id, answer)
        print(f"[Bot]: {answer[:100]}...")

    async def run(self):
        """Start the bot."""
        await self.login()
        self.client.add_event_callback(self.on_invite, MatrixInvitedRoom)
        self.client.add_event_callback(self.on_message, RoomMessageText)

        print("Syncing...")
        await self.client.sync(timeout=30000, full_state=True)
        for room_id, room in self.client.rooms.items():
            print(f"  In room: {room.display_name}")

        print("Listening for messages...")
        await self.client.sync_forever(timeout=30000)


def main():
    """CLI: email-matrix"""
    asyncio.run(EmailBot().run())


if __name__ == "__main__":
    main()
