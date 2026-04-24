# hitl_module.py
# ─────────────────────────────────────────────────────────────
# Human-in-the-Loop (HITL) escalation module
# Detects when to escalate and integrates human responses
# ─────────────────────────────────────────────────────────────

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from colorama import Fore, Style, init

import config

init(autoreset=True)
logger = logging.getLogger(__name__)


# ── Escalation Reasons (enum-style constants) ─────────────────
class EscalationReason:
    LOW_CONFIDENCE   = "low_confidence"       # LLM not sure of answer
    NO_CONTEXT       = "no_context_found"     # vector search returned nothing
    COMPLEX_QUERY    = "complex_query"        # multi-step / sensitive topic
    USER_REQUESTED   = "user_requested"       # user explicitly asked for human
    LLM_FAILURE      = "llm_failure"          # LLM threw an exception


# ── HITL Module ────────────────────────────────────────────────

class HITLModule:
    """
    Manages Human-in-the-Loop escalation.

    Responsibilities:
    1. Detect escalation triggers (confidence, keywords, intent)
    2. Prompt human agent (CLI in this implementation; swap for API/webhook in prod)
    3. Log all escalation events to a JSONL file for audit / analysis
    4. Return human response back into the graph state
    """

    def __init__(self, log_file: str = config.HITL_LOG_FILE):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"HITL module ready | log={self.log_file}")

    # ── Escalation Detection ───────────────────────────────────

    def should_escalate(
        self,
        query:       str,
        answer:      str,
        confidence:  float,
        context_found: bool,
        intent:      str,
    ) -> tuple[bool, str]:
        """
        Returns (should_escalate: bool, reason: str).

        Decision tree:
        1. User explicitly asked for a human → escalate
        2. No context was retrieved → escalate (can't answer from KB)
        3. Confidence below threshold → escalate
        4. Intent is billing/legal (sensitive) with low context → escalate
        """
        query_lower = query.lower()

        # Rule 1: User explicitly requested a human
        for phrase in config.ESCALATION_PHRASES:
            if phrase in query_lower:
                return True, EscalationReason.USER_REQUESTED

        # Rule 2: No supporting context found in knowledge base
        if not context_found:
            return True, EscalationReason.NO_CONTEXT

        # Rule 3: Low confidence score from LLM
        if confidence < config.CONFIDENCE_THRESHOLD:
            return True, EscalationReason.LOW_CONFIDENCE

        # Rule 4: Sensitive intent + borderline confidence
        sensitive_intents = {"billing", "legal", "escalation"}
        if intent in sensitive_intents and confidence < 0.7:
            return True, EscalationReason.COMPLEX_QUERY

        return False, ""

    # ── Human Interaction ──────────────────────────────────────

    def request_human_response(
        self,
        query:   str,
        reason:  str,
        context: list,
        auto_response: Optional[str] = None,
    ) -> str:
        """
        In a real production system this would:
        - Send ticket to Zendesk / Freshdesk
        - Ping Slack channel
        - Add to human agent queue

        Here we use CLI interaction for demonstration.
        Set `auto_response` for automated testing.
        """
        print(f"\n{Fore.YELLOW}{'─'*60}")
        print(f"{Fore.YELLOW}⚠  HITL ESCALATION TRIGGERED")
        print(f"{Fore.YELLOW}{'─'*60}")
        print(f"{Fore.CYAN}Reason    : {reason}")
        print(f"{Fore.CYAN}User Query: {query}")
        if context:
            print(f"{Fore.CYAN}Context   : {context[0][:200]}...")
        print(f"{Fore.YELLOW}{'─'*60}")
        print(f"{Fore.GREEN}[Human Agent] Please type your response below:")
        print(f"{Fore.WHITE}(or press Enter to use default message){Style.RESET_ALL}")

        if auto_response is not None:
            human_response = auto_response
            print(f"[AUTO] {human_response}")
        else:
            human_response = input("Agent Response → ").strip()

        if not human_response:
            human_response = (
                "Thank you for reaching out. A support agent has been notified "
                "and will contact you within 24 hours via email."
            )

        self._log_event(query, reason, human_response, context)
        return human_response

    # ── Logging ────────────────────────────────────────────────

    def _log_event(
        self,
        query:          str,
        reason:         str,
        human_response: str,
        context:        list,
    ):
        """Append escalation event to JSONL log (one JSON object per line)."""
        event = {
            "timestamp":      datetime.utcnow().isoformat() + "Z",
            "query":          query,
            "escalation_reason": reason,
            "human_response": human_response,
            "context_count":  len(context),
        }
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        logger.info(f"HITL event logged → {self.log_file}")

    def get_escalation_stats(self) -> dict:
        """Parse the log file and return aggregate stats."""
        if not self.log_file.exists():
            return {"total_escalations": 0}

        events = []
        with self.log_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        reasons = {}
        for e in events:
            r = e.get("escalation_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1

        return {
            "total_escalations": len(events),
            "by_reason":         reasons,
        }
