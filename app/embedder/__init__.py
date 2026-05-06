from typing import TYPE_CHECKING

from app.embedder.embedder_base import BaseEmbedder

if TYPE_CHECKING:  # pragma: no cover
    from app.embedder.bge_embedder import BGEEmbedder as BGEEmbedder  # noqa: F401
    from app.embedder.embedder_factory import embed_chunks as embed_chunks  # noqa: F401
    from app.embedder.embedder_factory import get_embedder as get_embedder  # noqa: F401
    from app.embedder.minilm_embedder import (  # noqa: F401
        SentenceTransformerEmbedder as SentenceTransformerEmbedder,
    )

__all__ = [
    "BaseEmbedder",
    "BGEEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
    "embed_chunks",
]


def __getattr__(name: str):
    if name == "BGEEmbedder":
        from app.embedder.bge_embedder import BGEEmbedder

        return BGEEmbedder

    if name == "SentenceTransformerEmbedder":
        from app.embedder.minilm_embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder

    if name in {"get_embedder", "embed_chunks"}:
        from app.embedder.embedder_factory import embed_chunks, get_embedder

        return {
            "get_embedder": get_embedder,
            "embed_chunks": embed_chunks,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
