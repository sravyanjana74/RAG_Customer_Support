# main.py
# ─────────────────────────────────────────────────────────────
# Entry point for the RAG Customer Support System
# Run:  python main.py
# ─────────────────────────────────────────────────────────────

import os
import sys
import logging
from pathlib import Path
from colorama import Fore, Style, Back, init

import config
from document_processor import DocumentProcessor
from embedding_store import EmbeddingStore
from hitl_module import HITLModule
from graph_workflow import RAGWorkflow

init(autoreset=True)
logging.basicConfig(
    level  = logging.WARNING,          # suppress debug noise in CLI
    format = "%(levelname)s | %(name)s | %(message)s",
)


# ── Banner ─────────────────────────────────────────────────────

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════╗
║   RAG CUSTOMER SUPPORT ASSISTANT  v1.0               ║
║   Powered by LangGraph + ChromaDB + LLM              ║
╚══════════════════════════════════════════════════════╝{Style.RESET_ALL}
Commands:
  {Fore.GREEN}[type your question]{Style.RESET_ALL}  → get an AI answer
  {Fore.YELLOW}ingest{Style.RESET_ALL}                → (re)load PDFs from ./knowledge_base
  {Fore.YELLOW}stats{Style.RESET_ALL}                 → show system stats
  {Fore.YELLOW}sources{Style.RESET_ALL}               → list indexed documents
  {Fore.YELLOW}hitl-stats{Style.RESET_ALL}            → show escalation statistics
  {Fore.YELLOW}reset-db{Style.RESET_ALL}              → clear vector store
  {Fore.RED}exit{Style.RESET_ALL}                  → quit
"""


# ── Ingestion Helper ───────────────────────────────────────────

def ingest_documents(store: EmbeddingStore, processor: DocumentProcessor):
    """Load PDFs from the knowledge_base folder into ChromaDB."""
    pdf_dir = Path(config.PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(
            f"{Fore.YELLOW}No PDFs found in '{config.PDF_DIR}/'.\n"
            f"  → Place your PDF files there and run 'ingest' again.{Style.RESET_ALL}"
        )
        return

    print(f"\n{Fore.CYAN}Ingesting {len(pdf_files)} PDF(s)...{Style.RESET_ALL}")
    chunks = processor.load_directory(config.PDF_DIR)
    stats  = processor.chunk_stats(chunks)
    print(
        f"  Split into {stats.get('total_chunks', 0)} chunks | "
        f"avg {stats.get('avg_chars', 0)} chars/chunk"
    )

    added = store.add_chunks(chunks)
    print(
        f"{Fore.GREEN}✓ Done. {added} new chunks stored. "
        f"Total in DB: {store.count()}{Style.RESET_ALL}\n"
    )


# ── Response Display ───────────────────────────────────────────

def display_response(state: dict):
    """Pretty-print the final graph state to the terminal."""
    print(f"\n{Fore.CYAN}{'─'*60}{Style.RESET_ALL}")

    intent     = state.get("intent", "?")
    confidence = state.get("confidence", 0.0) or 0.0
    escalated  = state.get("escalated", False)
    response   = state.get("final_response", "No response generated.")

    # Header info
    conf_color = Fore.GREEN if confidence >= 0.7 else (
        Fore.YELLOW if confidence >= 0.5 else Fore.RED
    )
    print(
        f"{Fore.WHITE}Intent: {Fore.MAGENTA}{intent}  "
        f"{Fore.WHITE}| Confidence: {conf_color}{confidence:.0%}  "
        f"{Fore.WHITE}| Escalated: "
        f"{'🔴 Yes' if escalated else '🟢 No'}{Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}{'─'*60}{Style.RESET_ALL}")

    if escalated:
        print(f"{Fore.YELLOW}🤝 {response}{Style.RESET_ALL}")
    else:
        print(f"{Fore.WHITE}{response}{Style.RESET_ALL}")

    print(f"{Fore.CYAN}{'─'*60}{Style.RESET_ALL}\n")


# ── Main Loop ──────────────────────────────────────────────────

def main():
    print(BANNER)

    # ── Initialize components ──────────────────────────────────
    print(f"{Fore.CYAN}Initializing system...{Style.RESET_ALL}")
    processor = DocumentProcessor()
    store     = EmbeddingStore()
    hitl      = HITLModule()
    workflow  = RAGWorkflow(store=store, hitl=hitl)
    print(f"{Fore.GREEN}✓ System ready. ChromaDB has {store.count()} chunks.{Style.RESET_ALL}")

    # Auto-ingest if DB is empty
    if store.count() == 0:
        print(f"\n{Fore.YELLOW}Vector store is empty. Attempting initial ingest...{Style.RESET_ALL}")
        ingest_documents(store, processor)

    print()

    # ── Chat Loop ──────────────────────────────────────────────
    while True:
        try:
            user_input = input(f"{Fore.GREEN}You → {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.RED}Bye!{Style.RESET_ALL}")
            break

        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────────────
        cmd = user_input.lower()

        if cmd in ("exit", "quit", "bye"):
            print(f"{Fore.RED}Goodbye!{Style.RESET_ALL}")
            break

        elif cmd == "ingest":
            ingest_documents(store, processor)
            # Rebuild workflow in case store changed
            workflow = RAGWorkflow(store=store, hitl=hitl)
            continue

        elif cmd == "stats":
            print(f"\n{Fore.CYAN}System Stats:{Style.RESET_ALL}")
            print(f"  Chunks in DB : {store.count()}")
            print(f"  Indexed files: {', '.join(store.get_sources()) or 'none'}")
            print(f"  LLM provider : {config.LLM_PROVIDER}")
            print(f"  Chunk size   : {config.CHUNK_SIZE} chars")
            print(f"  Top-K        : {config.TOP_K_RESULTS}")
            print(f"  Confidence   : ≥ {config.CONFIDENCE_THRESHOLD}\n")
            continue

        elif cmd == "sources":
            sources = store.get_sources()
            print(f"\n{Fore.CYAN}Indexed Sources:{Style.RESET_ALL}")
            for s in sources:
                print(f"  • {s}")
            print()
            continue

        elif cmd == "hitl-stats":
            stats = hitl.get_escalation_stats()
            print(f"\n{Fore.CYAN}HITL Statistics:{Style.RESET_ALL}")
            print(f"  Total escalations: {stats.get('total_escalations', 0)}")
            for reason, count in stats.get("by_reason", {}).items():
                print(f"  {reason}: {count}")
            print()
            continue

        elif cmd == "reset-db":
            confirm = input(
                f"{Fore.RED}This will delete all indexed data. "
                f"Type 'yes' to confirm: {Style.RESET_ALL}"
            ).strip().lower()
            if confirm == "yes":
                store.reset()
                print(f"{Fore.GREEN}Database cleared.{Style.RESET_ALL}\n")
            continue

        # ── RAG Query ──────────────────────────────────────────
        print(f"{Fore.CYAN}Thinking...{Style.RESET_ALL}")
        try:
            final_state = workflow.run(user_input)
            display_response(final_state)
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
