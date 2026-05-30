from pydantic import BaseModel


class OllamaConfig(BaseModel):
    host: str = "192.168.0.250"
    port: int = 11434

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class QdrantConfig(BaseModel):
    host: str = "192.168.0.250"
    port: int = 6333
    collection: str = "emails"


class GmailConfig(BaseModel):
    credentials_file: str = "data/credentials.json"
    token_file: str = "data/token.json"
    scopes: list[str] = ["https://www.googleapis.com/auth/gmail.readonly"]


class ModelsConfig(BaseModel):
    # Pre-processing
    text_llm: str = "qwen3:8b"
    vision_llm: str = "qwen2.5vl:7b"
    # Embedding
    embedding: str = "bge-m3:latest"
    embedding_dim: int = 1024
    # Chat (lightweight, always-on)
    chat: str = "gemma3:latest"


class AppConfig(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    qdrant: QdrantConfig = QdrantConfig()
    gmail: GmailConfig = GmailConfig()
    models: ModelsConfig = ModelsConfig()
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Max attachment size to process (bytes) — skip very large files
    max_attachment_size: int = 20 * 1024 * 1024  # 20MB
    # Max images per email to describe (skip rest to save time)
    max_images_per_email: int = 10


config = AppConfig()
