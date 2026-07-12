"""Text chunking utilities."""


def _find_split_end(text: str, start: int, maximum_end: int) -> int:
    """Find the best natural boundary within a chunk limit."""
    if maximum_end == len(text):
        return maximum_end

    for separator in ("\n\n", "\n", " "):
        boundary = text.rfind(separator, start + 1, maximum_end + 1)
        if boundary > start:
            return boundary + len(separator)
    return maximum_end


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text at natural boundaries with optional character overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        maximum_end = min(start + chunk_size, len(text))
        end = _find_split_end(text, start, maximum_end)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks
