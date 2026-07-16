"""Safe orchestration and deterministic fallback for agent planning."""

import logging
from collections.abc import Callable
from typing import Literal

from app.agent.planner import AgentPlanner, DeterministicAgentPlanner, LLMAgentPlanner
from app.agent.planner_models import AgentDecision, AgentPlanningResult
from app.agent.planner_parser import AgentPlanningError
from app.agent.planning_policy import AgentDecisionPolicy
from app.config import Settings
from app.services.llm_providers import LLMProvider, build_agent_planning_provider

PlanningStrategy = Literal["deterministic", "llm"]
PlannerFactory = Callable[[], AgentPlanner]
ProviderFactory = Callable[[], LLMProvider]
logger = logging.getLogger(__name__)

SAFE_PLANNING_ERROR_REASONS = {
    "empty_response": "LLM planner returned an empty response.",
    "invalid_json": "LLM planner returned invalid JSON.",
    "invalid_decision": "LLM planner returned an invalid structured decision.",
    "provider_failure": "LLM planner provider request failed.",
    "planning_error": "LLM planner failed to produce a valid decision.",
}


class AgentPlanningServiceError(RuntimeError):
    """Raised when no acceptable planning decision can be produced."""


class AgentPlanningService:
    """Choose one planner decision and safely coordinate deterministic fallback."""

    def __init__(
        self,
        primary_planner: AgentPlanner,
        fallback_planner: AgentPlanner,
        policy: AgentDecisionPolicy,
        requested_strategy: PlanningStrategy = "llm",
        fallback_enabled: bool = True,
    ) -> None:
        if requested_strategy not in {"deterministic", "llm"}:
            raise ValueError(f"Unsupported planning strategy: {requested_strategy}")
        self._primary_planner = primary_planner
        self._fallback_planner = fallback_planner
        self._policy = policy
        self._requested_strategy = requested_strategy
        self._fallback_enabled = fallback_enabled

    def create_plan(self, query: str) -> AgentPlanningResult:
        """Return one accepted decision without executing its tool steps."""
        logger.info(
            "agent_planning_started requested_strategy=%s",
            self._requested_strategy,
        )
        if self._requested_strategy == "deterministic":
            return self._create_deterministic_result(query)

        try:
            decision = self._primary_planner.create_plan(query)
        except AgentPlanningError as error:
            return self._fallback(
                query,
                SAFE_PLANNING_ERROR_REASONS.get(
                    error.code,
                    SAFE_PLANNING_ERROR_REASONS["planning_error"],
                ),
                primary_error=error,
            )
        except Exception as error:
            return self._fallback(
                query,
                "LLM planner failed unexpectedly.",
                primary_error=error,
            )

        try:
            policy_result = self._policy.evaluate(decision)
        except Exception as error:
            return self._fallback(
                query,
                "LLM decision policy evaluation failed.",
                primary_error=error,
            )
        if not policy_result.accepted:
            reason = policy_result.reason or (
                "LLM decision was rejected by the execution-readiness policy."
            )
            logger.info(
                "agent_planning_policy_rejected plan=%s confidence=%.2f",
                decision.selected_plan,
                decision.confidence,
            )
            return self._fallback(
                query,
                reason,
                primary_decision=decision,
                policy_rejection_reason=reason,
            )

        result = AgentPlanningResult(
            decision=decision,
            requested_strategy="llm",
            used_strategy="llm",
            fallback_used=False,
        )
        logger.info(
            "agent_planning_completed requested_strategy=llm used_strategy=llm "
            "plan=%s confidence=%.2f fallback_used=false",
            decision.selected_plan,
            decision.confidence,
        )
        return result

    def _create_deterministic_result(self, query: str) -> AgentPlanningResult:
        """Call only the configured deterministic primary planner."""
        try:
            decision = self._primary_planner.create_plan(query)
        except Exception as error:
            logger.info(
                "agent_planning_failed requested_strategy=deterministic "
                "error_type=%s",
                type(error).__name__,
            )
            raise AgentPlanningServiceError(
                "Deterministic agent planning failed"
            ) from error
        result = AgentPlanningResult(
            decision=decision,
            requested_strategy="deterministic",
            used_strategy="deterministic",
            fallback_used=False,
        )
        logger.info(
            "agent_planning_completed requested_strategy=deterministic "
            "used_strategy=deterministic plan=%s confidence=%.2f "
            "fallback_used=false",
            decision.selected_plan,
            decision.confidence,
        )
        return result

    def _fallback(
        self,
        query: str,
        reason: str,
        primary_error: Exception | None = None,
        primary_decision: AgentDecision | None = None,
        policy_rejection_reason: str | None = None,
    ) -> AgentPlanningResult:
        """Use one deterministic fallback or raise when it is unavailable."""
        primary_error_type = (
            type(primary_error).__name__ if primary_error is not None else None
        )
        if not self._fallback_enabled:
            logger.info(
                "agent_planning_failed requested_strategy=llm error_type=%s",
                primary_error_type or "PolicyRejection",
            )
            error = AgentPlanningServiceError(
                "LLM agent planning failed or was rejected and fallback is disabled"
            )
            if primary_error is not None:
                raise error from primary_error
            raise error

        try:
            decision = self._fallback_planner.create_plan(query)
        except Exception as fallback_error:
            logger.info(
                "agent_planning_failed requested_strategy=llm error_type=%s "
                "fallback_error_type=%s",
                primary_error_type or "PolicyRejection",
                type(fallback_error).__name__,
            )
            raise AgentPlanningServiceError(
                "LLM agent planning and deterministic fallback both failed"
            ) from fallback_error

        result = AgentPlanningResult(
            decision=decision,
            requested_strategy="llm",
            used_strategy="deterministic",
            fallback_used=True,
            fallback_reason=reason,
            primary_error_type=primary_error_type,
            policy_rejection_reason=policy_rejection_reason,
            primary_decision=primary_decision,
        )
        logger.info(
            "agent_planning_fallback_used requested_strategy=llm "
            "used_strategy=deterministic plan=%s confidence=%.2f "
            "primary_error_type=%s",
            decision.selected_plan,
            decision.confidence,
            primary_error_type or "PolicyRejection",
        )
        logger.info(
            "agent_planning_completed requested_strategy=llm "
            "used_strategy=deterministic plan=%s confidence=%.2f "
            "fallback_used=true",
            decision.selected_plan,
            decision.confidence,
        )
        return result


class _LazyAgentPlanner:
    """Build one planner on first use so provider configuration stays lazy."""

    def __init__(self, factory: PlannerFactory) -> None:
        self._factory = factory
        self._planner: AgentPlanner | None = None

    def create_plan(self, query: str) -> AgentDecision:
        """Build once, then delegate without executing any tools."""
        if self._planner is None:
            self._planner = self._factory()
        return self._planner.create_plan(query)


def build_agent_planning_service(
    settings: Settings,
    provider_factory: ProviderFactory | None = None,
) -> AgentPlanningService:
    """Build configured orchestration without wiring it to agent execution."""
    deterministic_planner = DeterministicAgentPlanner()
    policy = AgentDecisionPolicy(
        settings.agent_min_planning_confidence,
        settings.agent_max_steps,
    )
    if settings.agent_planning_mode == "deterministic":
        return AgentPlanningService(
            deterministic_planner,
            deterministic_planner,
            policy,
            requested_strategy="deterministic",
            fallback_enabled=settings.agent_planning_fallback_enabled,
        )

    def build_llm_planner() -> AgentPlanner:
        provider = (
            provider_factory()
            if provider_factory is not None
            else build_agent_planning_provider(settings)
        )
        return LLMAgentPlanner(provider)

    return AgentPlanningService(
        _LazyAgentPlanner(build_llm_planner),
        deterministic_planner,
        policy,
        requested_strategy="llm",
        fallback_enabled=settings.agent_planning_fallback_enabled,
    )
