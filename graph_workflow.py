# graph_workflow.py
# ─────────────────────────────────────────────────────────────
# LangGraph-based RAG workflow with conditional routing & HITL
# ─────────────────────────────────────────────────────────────

import logging
import re
from typing import TypedDict, Optional, List, Annotated

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

import config
from embedding_store import EmbeddingStore, RetrievedChunk
from hitl_module import HITLModule, EscalationReason

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. GRAPH STATE  (the data object flowing between nodes)
# ══════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    """
    Shared state passed between every node in the graph.
    All fields are optional so nodes only populate what they own.
    """
    # Input
    query:           str

    # After intent classification
    intent:          Optional[str]           # billing | technical | account | general | escalation
    classified:      Optional[bool]

    # After retrieval
    retrieved_chunks: Optional[List[dict]]   # serialized RetrievedChunk dicts
    context_text:    Optional[str]           # concatenated context for the LLM prompt
    context_found:   Optional[bool]

    # After generation
    answer:          Optional[str]
    confidence:      Optional[float]         # 0.0 – 1.0 self-assessed confidence

    # HITL
    escalated:       Optional[bool]
    escalation_reason: Optional[str]
    human_response:  Optional[str]

    # Final
    final_response:  Optional[str]
    error:           Optional[str]


# ══════════════════════════════════════════════════════════════
# 2. LLM FACTORY  (swappable backend)
# ══════════════════════════════════════════════════════════════

def get_llm():
    """Return configured LLM. Supports OpenAI and Ollama."""
    if config.LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
        logger.info(f"Using Ollama LLM: {config.OLLAMA_MODEL}")
        return Ollama(
            model       = config.OLLAMA_MODEL,
            base_url    = config.OLLAMA_BASE_URL,
            temperature = config.LLM_TEMPERATURE,
        )
    else:  # default: openai
        from langchain_openai import ChatOpenAI
        logger.info(f"Using OpenAI LLM: {config.OPENAI_MODEL}")
        return ChatOpenAI(
            model       = config.OPENAI_MODEL,
            temperature = config.LLM_TEMPERATURE,
            max_tokens  = config.LLM_MAX_TOKENS,
            api_key     = config.OPENAI_API_KEY,
        )


# ══════════════════════════════════════════════════════════════
# 3. RAG WORKFLOW CLASS
# ══════════════════════════════════════════════════════════════

class RAGWorkflow:
    """
    Builds and runs the LangGraph workflow.

    Graph structure:
    ─────────────────────────────────────────────────────────────
        [START]
           │
           ▼
      [classify_intent]          ← classifies user query intent
           │
           ▼
      [retrieve_context]         ← semantic search in ChromaDB
           │
           ▼
      [generate_answer]          ← LLM synthesizes answer
           │
           ▼ (conditional routing)
      ┌────┴──────────────┐
      │                   │
    [output_node]   [hitl_escalation]  ← human agent handles it
      │                   │
      └────────┬──────────┘
               │
             [END]
    ─────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        store:      EmbeddingStore,
        hitl:       HITLModule,
        auto_hitl:  Optional[str] = None,   # pre-fill human response (for testing)
    ):
        self.store     = store
        self.hitl      = hitl
        self.llm       = get_llm()
        self.auto_hitl = auto_hitl
        self.graph     = self._build_graph()

    # ── Node 1: Intent Classification ─────────────────────────

    def _classify_intent(self, state: GraphState) -> GraphState:
        """
        Lightweight keyword-based intent classifier.
        Could be replaced with an LLM call for higher accuracy.
        """
        query = state["query"].lower()
        detected_intent = "general"

        for intent, keywords in config.INTENT_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                detected_intent = intent
                break

        logger.info(f"Intent classified: {detected_intent}")
        return {**state, "intent": detected_intent, "classified": True}

    # ── Node 2: Context Retrieval ──────────────────────────────

    def _retrieve_context(self, state: GraphState) -> GraphState:
        """
        Semantic search against ChromaDB.
        Serializes results so they fit in the TypedDict state.
        """
        query   = state["query"]
        results = self.store.search(query, top_k=config.TOP_K_RESULTS)

        if not results:
            logger.warning("No relevant context found.")
            return {
                **state,
                "retrieved_chunks": [],
                "context_text":     "",
                "context_found":    False,
            }

        # Build numbered context block for the prompt
        context_parts = []
        for i, chunk in enumerate(results, 1):
            context_parts.append(
                f"[Source {i} | {chunk.metadata.get('source','?')} "
                f"p.{chunk.metadata.get('page','?')} | "
                f"score={chunk.score:.2f}]\n{chunk.content}"
            )
        context_text = "\n\n".join(context_parts)

        serialized = [
            {"content": c.content, "metadata": c.metadata, "score": c.score}
            for c in results
        ]

        logger.info(f"Retrieved {len(results)} chunks for query.")
        return {
            **state,
            "retrieved_chunks": serialized,
            "context_text":     context_text,
            "context_found":    True,
        }

    # ── Node 3: Answer Generation ──────────────────────────────

    def _generate_answer(self, state: GraphState) -> GraphState:
        """
        Calls the LLM with retrieved context to generate a grounded answer.
        Also asks the LLM to self-report its confidence (0.0–1.0).
        """
        query        = state["query"]
        context_text = state.get("context_text", "")
        intent       = state.get("intent", "general")

        system_prompt = f"""You are a professional customer support AI assistant.
You help users by answering questions strictly based on the provided knowledge base context.

Rules:
1. Only answer from the provided context — do NOT hallucinate.
2. If the context is insufficient, clearly say so.
3. Be concise, friendly, and professional.
4. At the very end of your response, on a new line, write:
   CONFIDENCE: <number between 0.0 and 1.0>
   where 1.0 = fully confident, 0.0 = no confidence.
5. Current detected intent: {intent}
"""
        if context_text:
            user_message = (
                f"Context from knowledge base:\n{context_text}\n\n"
                f"User question: {query}\n\n"
                f"Please answer based on the context above."
            )
        else:
            user_message = (
                f"User question: {query}\n\n"
                f"Note: No relevant context was found in the knowledge base. "
                f"Please inform the user politely and suggest escalation."
            )

        try:
            if config.LLM_PROVIDER == "openai":
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
                response = self.llm.invoke(messages)
                raw_text = response.content
            else:
                # Ollama (plain LLM)
                full_prompt = f"{system_prompt}\n\n{user_message}"
                raw_text = self.llm.invoke(full_prompt)

            # Parse confidence from response
            confidence = self._extract_confidence(raw_text)
            answer     = self._strip_confidence_line(raw_text)

            logger.info(f"Answer generated | confidence={confidence:.2f}")
            return {
                **state,
                "answer":     answer,
                "confidence": confidence,
                "error":      None,
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                **state,
                "answer":           "I encountered an error generating a response.",
                "confidence":       0.0,
                "error":            str(e),
                "escalation_reason": EscalationReason.LLM_FAILURE,
            }

    # ── Node 4: HITL Escalation ────────────────────────────────

    def _hitl_escalation(self, state: GraphState) -> GraphState:
        """
        Escalation node: hands query to human agent and integrates response.
        """
        reason   = state.get("escalation_reason", EscalationReason.LOW_CONFIDENCE)
        context  = [c["content"] for c in (state.get("retrieved_chunks") or [])]
        human_resp = self.hitl.request_human_response(
            query         = state["query"],
            reason        = reason,
            context       = context,
            auto_response = self.auto_hitl,
        )
        return {
            **state,
            "escalated":       True,
            "human_response":  human_resp,
            "final_response":  f"[Escalated to Human Agent]\n{human_resp}",
        }

    # ── Node 5: Output Node ────────────────────────────────────

    def _output_node(self, state: GraphState) -> GraphState:
        """
        Packages the final response. This node always runs last for non-escalated paths.
        """
        answer = state.get("answer", "I was unable to generate a response.")
        sources = []
        for chunk in (state.get("retrieved_chunks") or []):
            src = chunk["metadata"].get("source", "?")
            pg  = chunk["metadata"].get("page", "?")
            ref = f"{src} (p.{pg})"
            if ref not in sources:
                sources.append(ref)

        citation = ""
        if sources:
            citation = "\n\n📄 Sources: " + ", ".join(sources)

        return {
            **state,
            "escalated":      False,
            "final_response": answer + citation,
        }

    # ── Conditional Router ─────────────────────────────────────

    def _route_after_generation(self, state: GraphState) -> str:
        """
        Decides next node after answer generation.
        Returns node name: "hitl_escalation" or "output_node"
        """
        # Explicit LLM failure
        if state.get("error") and "llm_failure" in (state.get("escalation_reason") or ""):
            logger.info("Routing → HITL (LLM failure)")
            return "hitl_escalation"

        escalate, reason = self.hitl.should_escalate(
            query         = state["query"],
            answer        = state.get("answer", ""),
            confidence    = state.get("confidence", 0.0),
            context_found = state.get("context_found", False),
            intent        = state.get("intent", "general"),
        )

        if escalate:
            logger.info(f"Routing → HITL ({reason})")
            # Inject reason into state via a workaround
            # (In LangGraph, we mutate via return value)
            state["escalation_reason"] = reason
            return "hitl_escalation"

        logger.info("Routing → Output Node")
        return "output_node"

    # ── Graph Builder ──────────────────────────────────────────

    def _build_graph(self):
        """Constructs the LangGraph StateGraph."""
        builder = StateGraph(GraphState)

        # Register nodes
        builder.add_node("classify_intent",  self._classify_intent)
        builder.add_node("retrieve_context", self._retrieve_context)
        builder.add_node("generate_answer",  self._generate_answer)
        builder.add_node("hitl_escalation",  self._hitl_escalation)
        builder.add_node("output_node",      self._output_node)

        # Define edges (control flow)
        builder.set_entry_point("classify_intent")
        builder.add_edge("classify_intent",  "retrieve_context")
        builder.add_edge("retrieve_context", "generate_answer")

        # Conditional routing AFTER generation
        builder.add_conditional_edges(
            source         = "generate_answer",
            path           = self._route_after_generation,
            path_map       = {
                "hitl_escalation": "hitl_escalation",
                "output_node":     "output_node",
            },
        )

        # Both terminal nodes go to END
        builder.add_edge("hitl_escalation", END)
        builder.add_edge("output_node",     END)

        return builder.compile()

    # ── Public API ─────────────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Execute the full RAG pipeline for a user query.
        Returns the final GraphState as a dict.
        """
        logger.info(f"Graph run started for query: '{query}'")
        initial_state: GraphState = {
            "query":             query,
            "intent":            None,
            "classified":        False,
            "retrieved_chunks":  None,
            "context_text":      None,
            "context_found":     False,
            "answer":            None,
            "confidence":        None,
            "escalated":         False,
            "escalation_reason": None,
            "human_response":    None,
            "final_response":    None,
            "error":             None,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Parse 'CONFIDENCE: 0.85' from LLM output."""
        match = re.search(r"CONFIDENCE:\s*([0-9.]+)", text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.75   # default if LLM didn't include it

    @staticmethod
    def _strip_confidence_line(text: str) -> str:
        """Remove the CONFIDENCE line from the answer shown to user."""
        return re.sub(r"\n?CONFIDENCE:\s*[0-9.]+\s*$", "", text, flags=re.IGNORECASE).strip()
