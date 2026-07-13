"""Thin Streamlit interface for indexing and grounded questions."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from app.agent import AgentResponse, build_agent
from app.config import ConfigurationError, Settings, get_settings
from app.services.answering import AnswerGenerationError, AnswerResult
from app.services.embeddings import EmbeddingError
from app.services.llm_providers import LLMProviderError
from app.services.retrieval import RetrievalResult
from app.services.routing import RoutingDecision
from app.services.runtime import (
    RoutedAnswer,
    RoutedSearch,
    answer_with_routing,
    build_collection_registry,
    build_index_services,
)
from app.services.uploads import UploadData, index_uploads
from app.services.vector_store import VectorStoreError


def main() -> None:
    """Render the MiniRAG Streamlit application."""
    st.set_page_config(page_title="MiniRAG Assistant", page_icon="📚")
    st.title("MiniRAG Assistant")
    st.caption(
        "Index local documents and ask grounded questions with source citations."
    )

    try:
        settings = get_settings()
    except ConfigurationError as error:
        st.error(str(error))
        return

    with st.sidebar:
        st.header("Documents")
        registry = build_collection_registry(settings)
        selected_collection = st.selectbox(
            "Indexing collection",
            options=registry.list_collections(),
            index=registry.list_collections().index(registry.default_collection),
        )
        query_mode_label = st.radio(
            "Query mode",
            options=("Manual collection", "Automatic routing"),
            index=0 if settings.default_query_mode == "manual" else 1,
        )
        automatic_routing = query_mode_label == "Automatic routing"
        use_agent = st.checkbox("Use Agent", value=False)
        if use_agent:
            st.caption(
                "The agent selects one tool. Uploads still use the indexing "
                f"collection **{selected_collection}**."
            )
        elif automatic_routing:
            st.caption(
                "Questions are routed to one collection. Uploads still use "
                f"**{selected_collection}**."
            )
        else:
            st.caption(f"Selected collection: **{selected_collection}**")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or Markdown",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("Index documents", disabled=not uploaded_files):
            _index_uploaded_files(uploaded_files, settings, selected_collection)
        st.divider()
        st.subheader("Configuration")
        st.text(f"LLM provider: {settings.llm_provider}")
        st.text(f"LLM model: {settings.llm_model}")
        st.text(f"Embedding: {settings.embedding_model}")
        st.caption(
            "Retrieved text chunks are sent to the selected external LLM API. "
            "API keys are never displayed."
        )

    question = st.text_input("Ask a question about indexed documents")
    if st.button("Ask", disabled=not question.strip(), type="primary"):
        try:
            outcome = _run_question(
                question,
                settings,
                selected_collection,
                automatic_routing,
                use_agent,
            )
        except (
            AnswerGenerationError,
            EmbeddingError,
            LLMProviderError,
            VectorStoreError,
            ValueError,
        ) as error:
            st.error(str(error))
        else:
            if isinstance(outcome, AgentResponse):
                _render_agent_response(outcome)
            else:
                _render_routed_answer(outcome, automatic_routing)


def _run_question(
    question: str,
    settings: Settings,
    selected_collection: str,
    automatic_routing: bool,
    use_agent: bool,
) -> AgentResponse | RoutedAnswer:
    """Keep normal answering unchanged unless agent mode is selected."""
    if use_agent:
        return build_agent(settings).run(question)
    return answer_with_routing(
        question,
        settings.default_top_k,
        settings,
        "automatic" if automatic_routing else "manual",
        None if automatic_routing else selected_collection,
    )


def _render_agent_response(response: AgentResponse) -> None:
    """Display one-step behavior or each result in a bounded plan."""
    if len(response.steps) == 1:
        st.info(
            f"Selected tool: **{response.decision.tool}**  \n"
            f"Reason: {response.decision.reason}"
        )
        _render_agent_tool_result(response.result, show_routing=True)
        return

    plan = response.plan
    if plan is None:
        st.error("Agent response is missing its execution plan")
        return
    st.info(f"Plan: **{plan.name}**  \nReason: {plan.reason}")
    for index, step in enumerate(response.steps, start=1):
        st.subheader(f"Step {index}: {step.tool}")
        _render_agent_tool_result(
            step.result, show_routing=step.tool == "routing"
        )


def _render_agent_tool_result(result: object, show_routing: bool) -> None:
    """Display one safe structured tool result."""
    if isinstance(result, RoutedAnswer):
        _render_routed_answer(result, show_routing)
    elif isinstance(result, RoutedSearch):
        if show_routing:
            _render_routing(result.routing)
        st.subheader("Search results")
        if not result.response.results:
            st.write("No relevant results found.")
        for match in result.response.results:
            _render_search_result(match)
    elif isinstance(result, RoutingDecision):
        _render_routing(result)
    else:
        st.subheader("Configured collections")
        for collection in result:
            st.markdown(f"- `{collection}`")


def _render_routed_answer(routed: RoutedAnswer, show_routing: bool) -> None:
    """Display an existing grounded answer without changing its content."""
    if show_routing:
        _render_routing(routed.routing)
    _render_answer(routed.answer)


def _render_routing(decision: RoutingDecision) -> None:
    """Display safe routing metadata."""
    details = (
        f"Selected collection: **{decision.collection}**  \n"
        f"Strategy: **{decision.strategy}**  \n"
        f"Reason: {decision.reason}"
    )
    if decision.confidence is not None:
        details += f"  \nConfidence: {decision.confidence:.2f}"
    st.info(details)


def _render_answer(result: AnswerResult) -> None:
    """Display an answer and its existing source presentation."""
    st.subheader("Answer")
    st.write(result.answer)
    if result.sources:
        with st.expander("Sources", expanded=True):
            for source in result.sources:
                page = f" · Page {source.page_number}" if source.page_number else ""
                st.markdown(
                    f"**[{source.label}] {Path(source.source_file).name}**"
                    f"{page}  \n"
                    f"Chunk {source.chunk_index} · "
                    f"Distance {source.distance:.4f}"
                )
                st.caption(source.text_preview)


def _render_search_result(result: RetrievalResult) -> None:
    """Display one retrieved chunk with source metadata."""
    page = f" · Page {result.page_number}" if result.page_number else ""
    st.markdown(
        f"**{Path(result.source_file).name}**{page}  \n"
        f"Chunk {result.chunk_index} · Distance {result.distance:.4f}"
    )
    st.caption(result.text)


def _index_uploaded_files(
    uploaded_files: Sequence[Any],
    settings: Settings,
    collection: str | None = None,
) -> None:
    """Convert Streamlit uploads and delegate indexing to shared services."""
    try:
        files = [
            UploadData(filename=uploaded.name, content=uploaded.getvalue())
            for uploaded in uploaded_files
        ]
        registry = build_collection_registry(settings)
        logical_collection = registry.resolve_logical_name(collection)
        embedder, vector_store = build_index_services(
            settings, logical_collection
        )
        results = index_uploads(
            uploads=files,
            upload_directory=settings.upload_dir,
            max_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embedder=embedder,
            vector_store=vector_store,
            collection=logical_collection,
        )
    except (EmbeddingError, VectorStoreError, OSError, ValueError) as error:
        st.error(str(error))
        return
    for result in results:
        message = (
            f"{result.filename}: {result.status} in {result.collection}"
        )
        if result.status == "failed":
            st.error(f"{message} — {result.error}")
        elif result.status == "already_indexed":
            st.info(message)
        else:
            st.success(f"{message} ({result.stored_chunks} chunk(s))")


if __name__ == "__main__":
    main()
