# test_system.py
# ─────────────────────────────────────────────────────────────
# Tests for all RAG system components
# Run:  python test_system.py
# ─────────────────────────────────────────────────────────────

import os
import sys
import json
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((status, label))
    print(f"  {status}  {label}" + (f" → {detail}" if detail else ""))
    return condition


# ══════════════════════════════════════════════════════════════
print("\n📋 TEST 1: Config Module")
# ══════════════════════════════════════════════════════════════
try:
    import config
    check("CHUNK_SIZE is int",          isinstance(config.CHUNK_SIZE, int))
    check("CHUNK_OVERLAP < CHUNK_SIZE", config.CHUNK_OVERLAP < config.CHUNK_SIZE)
    check("TOP_K_RESULTS > 0",          config.TOP_K_RESULTS > 0)
    check("CONFIDENCE_THRESHOLD 0-1",   0 < config.CONFIDENCE_THRESHOLD < 1)
    check("INTENT_KEYWORDS populated",  len(config.INTENT_KEYWORDS) > 0)
except Exception as e:
    check("Config import", False, str(e))


# ══════════════════════════════════════════════════════════════
print("\n📋 TEST 2: Document Processor")
# ══════════════════════════════════════════════════════════════
try:
    from document_processor import DocumentProcessor, DocumentChunk
    proc = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    check("DocumentProcessor created", True)

    # Test chunking via splitter directly
    sample_text = "This is a test sentence. " * 50  # ~1250 chars
    chunks = proc.splitter.split_text(sample_text)
    check("Text splits correctly",   len(chunks) > 1,          f"{len(chunks)} chunks")
    check("Chunks are strings",      all(isinstance(c, str) for c in chunks))

    # Test DocumentChunk dataclass
    dc = DocumentChunk(
        chunk_id="abc123", source_file="test.pdf",
        page_number=1, content="Hello world test content here",
        char_count=0,
    )
    check("DocumentChunk char_count auto-set", dc.char_count == len(dc.content))

    # Test chunk_stats on empty list
    stats = proc.chunk_stats([])
    check("chunk_stats empty list returns dict", isinstance(stats, dict))

    # Test missing directory handled
    chunks_empty = proc.load_directory("/tmp/nonexistent_rag_test_dir")
    check("Missing directory returns empty list", chunks_empty == [])

except Exception as e:
    check("Document processor tests", False, str(e))


# ══════════════════════════════════════════════════════════════
print("\n📋 TEST 3: Embedding Store (ChromaDB)")
# ══════════════════════════════════════════════════════════════
try:
    from document_processor import DocumentChunk
    from embedding_store import EmbeddingStore, RetrievedChunk

    with tempfile.TemporaryDirectory() as tmpdir:
        store = EmbeddingStore(
            persist_dir = tmpdir,
            collection  = "test_col",
            model_name  = config.EMBEDDING_MODEL,
        )
        check("EmbeddingStore created",   True)
        check("Initial count is 0",       store.count() == 0)

        # Embed a query
        emb = store.embed_text("test query")
        check("Embedding is a list",      isinstance(emb, list))
        check("Embedding has 384 dims",   len(emb) == 384, f"{len(emb)} dims")

        # Add chunks
        test_chunks = [
            DocumentChunk(
                chunk_id=f"id{i}", source_file="test.pdf",
                page_number=i, content=f"Sample support document content number {i}. "
                                        f"This talks about product features and billing.",
                char_count=0,
                metadata={"source": "test.pdf", "page": i},
            )
            for i in range(1, 6)
        ]
        added = store.add_chunks(test_chunks)
        check("add_chunks adds 5 chunks",   added == 5)
        check("count() is now 5",           store.count() == 5)

        # Test deduplication
        added2 = store.add_chunks(test_chunks)
        check("Duplicate add returns 0",    added2 == 0)

        # Search
        results_found = store.search("billing product features", top_k=3, min_score=0.0)
        check("Search returns results",     len(results_found) > 0)
        check("Results are RetrievedChunk", all(isinstance(r, RetrievedChunk) for r in results_found))
        check("Scores are 0-1",             all(0.0 <= r.score <= 1.0 for r in results_found))

        # Search on empty-ish threshold
        results_none = store.search("zxqwerty gibberish query abc", top_k=3, min_score=0.99)
        check("High-threshold returns nothing", len(results_none) == 0)

        # get_sources
        sources = store.get_sources()
        check("get_sources returns list",   isinstance(sources, list))
        check("Source 'test.pdf' found",    "test.pdf" in sources)

        # Reset
        store.reset()
        check("reset() empties store",     store.count() == 0)

except Exception as e:
    check("Embedding store tests", False, str(e))
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════
print("\n📋 TEST 4: HITL Module")
# ══════════════════════════════════════════════════════════════
try:
    from hitl_module import HITLModule, EscalationReason

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = str(Path(tmpdir) / "test_hitl.jsonl")
        hitl = HITLModule(log_file=log_path)
        check("HITLModule created", True)

        # should_escalate: user request
        esc, reason = hitl.should_escalate(
            query="I want to speak to a human", answer="", confidence=0.9,
            context_found=True, intent="general"
        )
        check("Detects user-requested escalation",  esc, reason)

        # should_escalate: no context
        esc2, reason2 = hitl.should_escalate(
            query="How do I configure X?", answer="", confidence=0.9,
            context_found=False, intent="technical"
        )
        check("Escalates on no context",            esc2, reason2)

        # should_escalate: low confidence
        esc3, reason3 = hitl.should_escalate(
            query="What is refund policy?", answer="...", confidence=0.3,
            context_found=True, intent="billing"
        )
        check("Escalates on low confidence",        esc3, reason3)

        # should_escalate: NOT triggered (normal case)
        esc4, _ = hitl.should_escalate(
            query="How do I reset my password?", answer="Click Settings > Account.",
            confidence=0.88, context_found=True, intent="account"
        )
        check("Does NOT escalate high-conf answer", not esc4)

        # request_human_response with auto-response
        resp = hitl.request_human_response(
            query="Help!", reason=EscalationReason.NO_CONTEXT,
            context=["Some context text"],
            auto_response="A human agent will contact you."
        )
        check("Human response returned",            resp == "A human agent will contact you.")

        # Log written
        check("Log file created",                   Path(log_path).exists())
        with open(log_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        check("Log has 1 entry",                    len(lines) == 1)
        check("Log entry has timestamp",            "timestamp" in lines[0])

        # Stats
        stats = hitl.get_escalation_stats()
        check("get_escalation_stats works",         stats["total_escalations"] == 1)

except Exception as e:
    check("HITL module tests", False, str(e))
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════
print("\n📋 TEST 5: Graph Workflow (LangGraph)")
# ══════════════════════════════════════════════════════════════
try:
    from embedding_store import EmbeddingStore
    from hitl_module import HITLModule
    from graph_workflow import RAGWorkflow, GraphState

    with tempfile.TemporaryDirectory() as tmpdir:
        store = EmbeddingStore(persist_dir=tmpdir, collection="wf_test")
        hitl  = HITLModule(log_file=str(Path(tmpdir) / "hitl.jsonl"))

        # Mock LLM so we don't need API keys in tests
        class MockLLM:
            def invoke(self, _):
                class Resp:
                    content = "The password reset link is available in Settings > Account.\nCONFIDENCE: 0.9"
                return Resp()

        workflow = RAGWorkflow(store=store, hitl=hitl, auto_hitl="Test human response")
        workflow.llm = MockLLM()
        check("RAGWorkflow created with mock LLM", True)

        # Seed some data
        from document_processor import DocumentChunk
        store.add_chunks([
            DocumentChunk(
                chunk_id="wf1", source_file="faq.pdf", page_number=1,
                content="To reset your password, go to Settings > Account > Reset Password. "
                        "An email will be sent to your registered address.",
                char_count=0, metadata={"source": "faq.pdf", "page": 1}
            )
        ])

        # Test intent classification node
        state: GraphState = {
            "query": "I forgot my password, how do I reset it?",
            "intent": None, "classified": False,
            "retrieved_chunks": None, "context_text": None, "context_found": False,
            "answer": None, "confidence": None,
            "escalated": False, "escalation_reason": None,
            "human_response": None, "final_response": None, "error": None,
        }
        after_classify = workflow._classify_intent(state)
        check("Intent classified as 'account'", after_classify["intent"] == "account",
              after_classify["intent"])

        # Test retrieval node
        after_retrieve = workflow._retrieve_context(after_classify)
        check("Context retrieved",              after_retrieve.get("context_found", False))
        check("Retrieved chunks is list",       isinstance(after_retrieve.get("retrieved_chunks"), list))

        # Test confidence extraction
        score = RAGWorkflow._extract_confidence("Here is the answer.\nCONFIDENCE: 0.87")
        check("Confidence parsed correctly",    abs(score - 0.87) < 0.001, str(score))

        stripped = RAGWorkflow._strip_confidence_line("Some answer.\nCONFIDENCE: 0.87")
        check("Confidence line stripped",       "CONFIDENCE" not in stripped)

        # Test escalation path: no context → HITL
        empty_state = {**state, "query": "speak to human please", "context_found": True,
                       "confidence": 0.9, "answer": "Some answer", "intent": "general",
                       "retrieved_chunks": []}
        route = workflow._route_after_generation({
            **empty_state,
            "query": "speak to a human",
            "confidence": 0.9,
            "context_found": True,
            "intent": "general",
            "escalation_reason": None,
        })
        check("Escalation phrase routes to HITL", route == "hitl_escalation", route)

        # Normal answer routes to output
        route2 = workflow._route_after_generation({
            **empty_state,
            "query": "how to reset password",
            "confidence": 0.9,
            "context_found": True,
            "intent": "account",
            "escalation_reason": None,
        })
        check("High-confidence routes to output", route2 == "output_node", route2)

except Exception as e:
    check("Graph workflow tests", False, str(e))
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
total   = len(results)
passed  = sum(1 for r, _ in results if r == PASS)
failed  = total - passed

print(f"\n{'═'*50}")
print(f"  Results: {passed}/{total} tests passed")
if failed:
    print(f"  FAILED:")
    for status, label in results:
        if status == FAIL:
            print(f"    • {label}")
print(f"{'═'*50}\n")
sys.exit(0 if failed == 0 else 1)
