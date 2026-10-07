"""Responsible-AI guardrails: no buy / do-not-buy advice, always with a disclaimer."""

from __future__ import annotations

import re

ADVICE_QUESTION_RE = re.compile(
    r"\b(should\s+i|do\s+i\s+need\s+to|is\s+it\s+(?:worth|wise|good)|can\s+you\s+(?:recommend|advise))\b.*"
    r"\b(buy|purchase|take|switch|port|cancel|renew|migrate|choose|go\s+with|worth)\b"
    r"|\b(buy|purchase)\s+or\s+not\b|\bwhich\s+(?:one\s+)?(?:should|is\s+better\s+to)\s+(?:i\s+)?(?:buy|choose)\b",
    re.I,
)
DIRECTIVE_RE = re.compile(
    r"(you\s+(?:should|must|ought\s+to|need\s+to)\s+(?:not\s+)?(?:buy|purchase|switch|cancel|renew|take)\b"
    r"|i\s+(?:strongly\s+)?(?:recommend|advise|suggest)\s+(?:that\s+you\s+)?(?:buy|buying|switch|switching|cancel|cancelling|not\s+buy)"
    r"|\bdo\s+not\s+buy\b|\bdon'?t\s+buy\b|\bbuy\s+this\s+policy\b|\bbest\s+to\s+(?:buy|switch|cancel)\b"
    r"|\bgo\s+ahead\s+and\s+(?:buy|switch)\b)",
    re.I,
)


def is_advice_request(question: str) -> bool:
    return bool(ADVICE_QUESTION_RE.search(question))


def contains_directive(text: str) -> bool:
    return bool(DIRECTIVE_RE.search(text))


def sanitize_output(text: str) -> str:
    """Neutralise any directive buy/switch language that an LLM may produce."""
    return DIRECTIVE_RE.sub("consider verifying the details of this policy", text)


def decision_support_reply(risk_summary: str | None = None) -> str:
    base = (
        "I can't tell you whether to buy, switch, renew or cancel a policy - that decision is yours, "
        "together with a licensed advisor or the insurer. What I can do is show where the policy "
        "looks stronger, where it needs caution, and which clauses are unclear."
    )
    if risk_summary:
        base += "\n\n" + risk_summary
    return base + "\n\nPlease confirm all points with the insurer or an advisor before deciding."
