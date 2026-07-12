"""Local sentence-transformer embedding service."""

from collections.abc import Sequence
from typing import Any, ClassVar


class EmbeddingError(RuntimeError):
    """Raised when local embedding generation fails."""


class EmbeddingService:
    """Lazily load and reuse a local sentence-transformer model."""

    _models: ClassVar[dict[str, Any]] = {}

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _get_model(self) -> Any:
        """Return a cached model, loading it only on first use."""
        if self.model_name not in self._models:
            try:
                from sentence_transformers import SentenceTransformer

                self._models[self.model_name] = SentenceTransformer(self.model_name)
            except Exception as error:
                raise EmbeddingError(
                    f"Could not load embedding model {self.model_name!r}"
                ) from error
        return self._models[self.model_name]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for document chunk texts."""
        if not texts:
            return []
        try:
            vectors = self._get_model().encode(
                list(texts), normalize_embeddings=True
            )
            return [[float(value) for value in vector] for vector in vectors]
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError("Could not embed document chunks") from error

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for one non-empty query."""
        if not query.strip():
            raise ValueError("Query must not be empty")
        return self.embed_documents([query])[0]
