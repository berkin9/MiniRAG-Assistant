"""Deterministic intent classification for agent tool selection."""

import re
from dataclasses import dataclass

from app.agent.models import Intent


@dataclass(frozen=True)
class IntentRule:
    """Regex patterns that identify one non-default intent."""

    intent: Intent
    patterns: tuple[str, ...]


INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule(
        Intent.COLLECTIONS,
        (
            r"\b(?:list|show|display)\s+(?:the\s+)?collections\b",
            r"\bwhat\s+(?:collections|are\s+the\s+collections)\b",
        ),
    ),
    IntentRule(
        Intent.ROUTING,
        (
            r"\bwhich\s+collection\b",
            r"\bwhy\s+(?:general|project|technical|policies)\b",
            r"\broute\s+(?:this|that|the\s+(?:question|request))\b",
        ),
    ),
    IntentRule(
        Intent.SEARCH,
        (
            r"\bsearch\b",
            r"\bfind\b",
            r"\bretrieve\b",
        ),
    ),
)


def classify_intent(request: str) -> Intent:
    """Return the first matching intent or default to grounded answering."""
    normalized = " ".join(request.casefold().split())
    for rule in INTENT_RULES:
        if any(re.search(pattern, normalized) for pattern in rule.patterns):
            return rule.intent
    return Intent.ASK
