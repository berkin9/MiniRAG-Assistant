"""Thin Streamlit interface for indexing and grounded questions."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from app.config import ConfigurationError, Settings, get_settings
from app.services.answering import AnswerGenerationError
from app.services.embeddings import EmbeddingError
from app.services.llm_providers import LLMProviderError
from app.services.runtime import (
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
        if automatic_routing:
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
            routed = answer_with_routing(
                question,
                settings.default_top_k,
                settings,
                "automatic" if automatic_routing else "manual",
                None if automatic_routing else selected_collection,
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
            result = routed.answer
            if automatic_routing:
                decision = routed.routing
                details = (
                    f"Selected collection: **{decision.collection}**  \n"
                    f"Strategy: **{decision.strategy}**  \n"
                    f"Reason: {decision.reason}"
                )
                if decision.confidence is not None:
                    details += f"  \nConfidence: {decision.confidence:.2f}"
                st.info(details)
            st.subheader("Answer")
            st.write(result.answer)
            if result.sources:
                with st.expander("Sources", expanded=True):
                    for source in result.sources:
                        page = (
                            f" · Page {source.page_number}"
                            if source.page_number
                            else ""
                        )
                        st.markdown(
                            f"**[{source.label}] {Path(source.source_file).name}**"
                            f"{page}  \n"
                            f"Chunk {source.chunk_index} · "
                            f"Distance {source.distance:.4f}"
                        )
                        st.caption(source.text_preview)


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
