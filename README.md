# Chat with Your Emails

Privacy-first email RAG pipeline. Everything runs locally — Gmail → LLM extraction → embeddings → Qdrant → chat via Element/Matrix or web UI.

## Architecture

```
Gmail API → Preprocess (qwen3:8b + qwen2.5vl:7b) → Embed (bge-m3) → Qdrant
                                                                           ↓
Element/Matrix Bot ←→ RAG Query ←→ gemma3:latest ←→ Qdrant search ←←←←←←┘
```

## Quick Start

### 1. Prerequisites

- **Ollama** running on your tower (port 11434) with models:
  - `qwen3:8b` — text extraction
  - `qwen2.5vl:7b` — image/document analysis
  - `bge-m3:latest` — embeddings
  - `gemma3:latest` — chat (lightweight, always-on)
- **Qdrant** running on your tower (port 6333)
- **Gmail OAuth credentials** — `data/credentials.json` from [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Credentials → OAuth 2.0 Client ID → Desktop app

### 2. Install

```bash
git clone git@github.com:7anand7s/chat-with-your-emails.git
cd chat-with-your-emails
pip install -e .
```

### 3. Gmail Setup (one-time)

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create project → Enable Gmail API
3. OAuth consent screen → Add your email as test user
4. Credentials → Create OAuth 2.0 Client ID → Desktop app
5. Download JSON → save as `data/credentials.json`

### 4. Run the Pipeline

```bash
# Fetch emails from Gmail
email-ingest --limit 100

# Preprocess (LLM/VLM extraction, saves to data/processed/)
email-preprocess --limit 50

# Embed into Qdrant
email-embed --limit 50

# Check progress
email-status
```

Each step is independent. Stop/resume anytime. Nothing gets redone.

### 5. Chat

**Option A: Element/Matrix (mobile-friendly)**
```bash
email-matrix  # Starts the bot
```
Then open Element app → connect to your Synapse server → invite @bot to a room → ask questions.

**Option B: Web API**
```bash
email-chat  # Starts FastAPI server on port 8501
```
Then POST to `http://localhost:8501/api/chat` with `{"message": "your question"}`.

**Option C: Jupyter Notebook**
Open `pipeline.ipynb` → run cells top to bottom.

## CLI Commands

| Command | Description |
|---------|-------------|
| `email-ingest [--limit N] [--query '...']` | Fetch emails from Gmail |
| `email-preprocess [--limit N]` | LLM/VLM extraction (concurrent, 4 workers) |
| `email-embed [--limit N]` | Chunk, embed, store in Qdrant |
| `email-chat` | Start web chat server (port 8501) |
| `email-matrix` | Start Matrix/Element bot |
| `email-status` | Show pipeline progress |
| `email-resume` | Resume paused pipeline |
| `email-reset` | Clear pipeline state |

## Pipeline Stages

```
fetched → cleaned → llm_extracted → vlm_processed → embedded → stored
```

- **fetched**: Raw email saved to `data/raw_emails/`
- **cleaned**: Body flagged (signatures, quotes, disclaimers marked, never deleted)
- **llm_extracted**: qwen3:8b extracts structured data (category, entities, action items, etc.)
- **vlm_processed**: qwen2.5vl:7b describes image attachments (skips logos/signatures)
- **embedded**: bge-m3 embeddings stored in Qdrant
- **stored**: Fully processed and searchable

## Preprocessing Details

### Body Cleaning
Never deletes data. Flags sections with roles and importance scores:
- `body` (importance 1.0) — main content
- `signature` (0.1) — email signatures
- `quoted_reply` (0.2) — quoted text
- `disclaimer` (0.05) — legal disclaimers
- `tracking` (0.0) — tracking pixels
- `unsubscribe` (0.05) — marketing footers

### Attachment Processing
- **Documents**: PDF (pdfplumber + page images), DOCX, XLSX, PPTX, ZIP, EML
- **Images**: VLM classifies first (skip logos/signatures/tracking pixels), then describes meaningful ones
- **Encrypted PDFs**: Set `PDF_PASSWORDS` env var (comma-separated)
- **Scanned PDFs**: Detected automatically, VLM does OCR

### LLM Extraction (qwen3:8b)
Extracts 25+ fields per email:
- Summary (5-6 sentences, no context loss)
- Category + subcategory (30+ categories)
- Entities: people, companies, products, locations, dates, monetary, account numbers
- Action items with assignee, deadline, priority
- Questions asked, decisions made, deadlines
- Financial info, sentiment, tone
- `context_for_future_queries` — synonyms for better search

### Concurrent Processing
Both preprocessing and embedding use `ThreadPoolExecutor` with 4 workers. Ollama handles concurrent requests via `OLLAMA_NUM_PARALLEL=4`.

## Matrix/Element Setup

### Server Components (on Unraid tower)
- **Synapse**: Matrix homeserver (port 8008)
- **Element Web**: Chat client (port 8075)

### Bot Configuration
- Bot account: `@bot:7anand7s.com`
- Login: Uses admin API (bypasses rate limits)
- Formatting: HTML (`<ul>/<li>/<strong>`) — no raw markdown

### Rate Limits
Synapse's `rc_login` config: `per_second: 0` means BLOCK (not unlimited). Set to 1000+ in `homeserver.yaml`.

### Starting the Bot
```bash
email-matrix
# Or permanently:
nohup email-matrix > /var/log/matrix-bot.log 2>&1 &
```

## Configuration

All config in `config/settings.py`:

```python
OllamaConfig: host="192.168.0.250", port=11434
QdrantConfig: host="192.168.0.250", port=6333, collection="emails"
ModelsConfig: text_llm="qwen3:8b", vision_llm="qwen2.5vl:7b",
              embedding="bge-m3:latest", chat="gemma3:latest"
```

## Project Structure

```
src/
├── ingestion/gmail_client.py      # Gmail API fetch
├── preprocessing/
│   ├── pipeline.py                # Main pipeline orchestration
│   ├── llm_processor.py           # qwen3:8b text extraction
│   ├── vlm_processor.py           # qwen2.5vl:7b image analysis
│   ├── body_cleaner.py            # Flag noise, never delete
│   └── document_extractor.py      # PDF/DOCX/XLSX/etc extraction
├── embedding/embedder.py          # bge-m3 embeddings
├── storage/vector_store.py        # Qdrant client
├── tracking/
│   ├── state.py                   # Pipeline state persistence
│   └── display.py                 # Progress display
├── chat/app.py                    # FastAPI web chat
├── matrix_bot.py                  # Matrix/Element bot
└── cli.py                         # CLI commands (status/resume/reset)
```

## Privacy

Everything runs locally:
- Emails never leave your tower
- All LLM inference via local Ollama
- Embeddings stored in local Qdrant
- Matrix Synapse is self-hosted
- No cloud services except Gmail API (for fetching only)

## License

Private project.
