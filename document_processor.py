# document_processor.py
# ─────────────────────────────────────────────────────────────
# Handles: PDF loading → text extraction → chunking
# ─────────────────────────────────────────────────────────────

import os
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """Represents a single processed chunk from a PDF document."""
    chunk_id:    str          # SHA-256 hash of content (deduplication key)
    source_file: str          # Original PDF filename
    page_number: int          # Page the chunk came from
    content:     str          # The actual text
    char_count:  int          # Character count
    metadata:    dict = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.content)
        if not self.chunk_id:
            self.chunk_id = hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ── Core Processor ────────────────────────────────────────────

class DocumentProcessor:
    """
    Loads PDFs and splits them into overlapping chunks ready for embedding.

    Strategy:
    - RecursiveCharacterTextSplitter splits on paragraph/sentence boundaries first,
      then falls back to character splits — preserving semantic coherence better
      than fixed-token splits.
    - Chunk overlap ensures context is not lost at boundaries.
    """

    def __init__(
        self,
        chunk_size: int    = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter      = RecursiveCharacterTextSplitter(
            chunk_size        = self.chunk_size,
            chunk_overlap     = self.chunk_overlap,
            length_function   = len,
            separators        = ["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            f"DocumentProcessor ready | chunk_size={chunk_size} "
            f"overlap={chunk_overlap}"
        )

    # ── Public API ─────────────────────────────────────────────

    def load_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Load a single PDF and return its chunks."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {pdf_path}")

        logger.info(f"Loading PDF: {path.name}")
        loader = PyPDFLoader(str(path))
        pages  = loader.load()                  # List[Document], one per page

        chunks: List[DocumentChunk] = []
        for page_doc in pages:
            page_num  = page_doc.metadata.get("page", 0) + 1   # 1-indexed
            raw_text  = page_doc.page_content.strip()
            if not raw_text:
                continue                                         # skip blank pages

            split_texts = self.splitter.split_text(raw_text)
            for text in split_texts:
                if len(text.strip()) < 30:                      # skip micro-chunks
                    continue
                chunk_id = hashlib.sha256(
                    (path.name + str(page_num) + text).encode()
                ).hexdigest()[:16]

                chunks.append(DocumentChunk(
                    chunk_id    = chunk_id,
                    source_file = path.name,
                    page_number = page_num,
                    content     = text.strip(),
                    char_count  = len(text.strip()),
                    metadata    = {
                        "source": path.name,
                        "page":   page_num,
                        "total_pages": len(pages),
                    },
                ))

        logger.info(f"  → {len(pages)} pages → {len(chunks)} chunks")
        return chunks

    def load_directory(self, directory: str = config.PDF_DIR) -> List[DocumentChunk]:
        """Load all PDFs from a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"PDF directory not found: {directory}. Creating it.")
            dir_path.mkdir(parents=True, exist_ok=True)
            return []

        pdf_files = list(dir_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in: {directory}")
            return []

        all_chunks: List[DocumentChunk] = []
        for pdf_file in pdf_files:
            try:
                chunks = self.load_pdf(str(pdf_file))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")

        logger.info(
            f"Loaded {len(pdf_files)} PDFs → {len(all_chunks)} total chunks"
        )
        return all_chunks

    def chunk_stats(self, chunks: List[DocumentChunk]) -> dict:
        """Return summary statistics about the chunks."""
        if not chunks:
            return {}
        char_counts = [c.char_count for c in chunks]
        return {
            "total_chunks":    len(chunks),
            "avg_chars":       round(sum(char_counts) / len(char_counts), 1),
            "min_chars":       min(char_counts),
            "max_chars":       max(char_counts),
            "unique_sources":  len({c.source_file for c in chunks}),
        }
