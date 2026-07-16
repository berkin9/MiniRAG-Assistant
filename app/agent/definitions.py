"""Central bounded-agent tool and plan definitions."""

from app.agent_limits import HARD_AGENT_MAX_STEPS

SUPPORTED_AGENT_TOOLS = frozenset({"ask", "search", "collections", "routing"})
SUPPORTED_AGENT_PLANS: dict[str, tuple[str, ...]] = {
    "ask": ("ask",),
    "search": ("search",),
    "collections": ("collections",),
    "routing": ("routing",),
    "route_and_ask": ("routing", "ask"),
    "route_and_search": ("routing", "search"),
}
