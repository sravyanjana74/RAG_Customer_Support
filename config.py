# config.py
# ─────────────────────────────────────────────────────────────
# Central configuration for the RAG Customer Support System
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Settings ──────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "openai")   # "openai" | "ollama"
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS      = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ── Embedding Settings ─────────────────────────────────────────
# Using sentence-transformers (free, runs locally — no API key needed)
EMBEDDING_MODEL     = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# ── ChromaDB Settings ─────────────────────────────────────────
CHROMA_PERSIST_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION   = os.getenv("CHROMA_COLLECTION", "support_kb")

# ── Document Processing ────────────────────────────────────────
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", "500"))       # tokens per chunk
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP", "100"))    # overlap between chunks
PDF_DIR             = os.getenv("PDF_DIR", "./knowledge_base")  # folder with PDFs

# ── Retrieval Settings ────────────────────────────────────────
TOP_K_RESULTS       = int(os.getenv("TOP_K_RESULTS", "4"))      # docs to retrieve
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))  # min score

# ── HITL Settings ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
HITL_LOG_FILE        = os.getenv("HITL_LOG_FILE", "./hitl_log.jsonl")

# ── Intent Keywords (extend as needed) ────────────────────────
INTENT_KEYWORDS = {
    "billing":      ["invoice", "payment", "charge", "bill", "refund", "price", "cost", "subscription"],
    "technical":    ["error", "bug", "crash", "not working", "issue", "problem", "broken", "fail"],
    "account":      ["login", "password", "account", "sign in", "access", "locked", "reset"],
    "general":      ["how", "what", "where", "when", "which", "help", "info", "tell me"],
    "escalation":   ["speak to human", "talk to agent", "human agent", "real person", "escalate", "urgent"],
}

# ── Escalation Triggers ────────────────────────────────────────
ESCALATION_PHRASES = [
    "speak to a human", "talk to someone", "real agent",
    "human support", "escalate", "not helpful", "useless",
    "emergency", "lawsuit", "legal action",
]
