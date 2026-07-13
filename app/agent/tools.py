"""Small adapters from agent tools to existing application services."""

from collections.abc import Callable
from typing import Protocol

from app.agent.models import AgentExecutionContext, AgentToolResult
from app.config import Settings
from app.services.routing import RoutingDecision
from app.services.runtime import (
    RoutedAnswer,
    RoutedSearch,
    answer_with_routing,
    build_collection_registry,
    route_with_settings,
    search_with_settings,
)

AskRunner = Callable[[str, int, Settings, str | None, str | None], RoutedAnswer]
SearchRunner = Callable[[str, int, Settings, str | None, str | None], RoutedSearch]
RoutingRunner = Callable[
    [str, Settings, str | None, str | None], RoutingDecision
]


class AgentTool(Protocol):
    """Common interface for one-shot internal agent tools."""

    name: str

    def run(self, request: str) -> AgentToolResult:
        """Handle one request and return a structured result."""


class AskTool:
    """Run the existing automatically routed grounded-answer pipeline."""

    name = "ask"

    def __init__(
        self, settings: Settings, runner: AskRunner = answer_with_routing
    ) -> None:
        self._settings = settings
        self._runner = runner

    def run(self, request: str) -> RoutedAnswer:
        """Answer through exactly one automatically selected collection."""
        return self.run_with_context(request, AgentExecutionContext())

    def run_with_context(
        self, request: str, context: AgentExecutionContext
    ) -> RoutedAnswer:
        """Reuse a prior routing selection when the bounded plan provides one."""
        mode = "manual" if context.selected_collection else "automatic"
        return self._runner(
            request,
            self._settings.default_top_k,
            self._settings,
            mode,
            context.selected_collection,
        )


class SearchTool:
    """Run routed semantic retrieval without answer generation."""

    name = "search"

    def __init__(
        self, settings: Settings, runner: SearchRunner = search_with_settings
    ) -> None:
        self._settings = settings
        self._runner = runner

    def run(self, request: str) -> RoutedSearch:
        """Retrieve chunks from exactly one automatically selected collection."""
        return self.run_with_context(request, AgentExecutionContext())

    def run_with_context(
        self, request: str, context: AgentExecutionContext
    ) -> RoutedSearch:
        """Reuse a prior routing selection when the bounded plan provides one."""
        mode = "manual" if context.selected_collection else "automatic"
        return self._runner(
            request,
            self._settings.default_top_k,
            self._settings,
            mode,
            context.selected_collection,
        )


class CollectionsTool:
    """Return configured logical collections without retrieval."""

    name = "collections"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, request: str) -> tuple[str, ...]:
        """List configured collections in registry order."""
        del request
        return build_collection_registry(self._settings).list_collections()


class RoutingTool:
    """Explain existing automatic routing without retrieval."""

    name = "routing"

    def __init__(
        self, settings: Settings, runner: RoutingRunner = route_with_settings
    ) -> None:
        self._settings = settings
        self._runner = runner

    def run(self, request: str) -> RoutingDecision:
        """Return the existing router's structured decision."""
        return self.run_with_context(request, AgentExecutionContext())

    def run_with_context(
        self, request: str, context: AgentExecutionContext
    ) -> RoutingDecision:
        """Record one routing decision in the per-run execution context."""
        decision = self._runner(request, self._settings, "automatic", None)
        context.selected_collection = decision.collection
        return decision
