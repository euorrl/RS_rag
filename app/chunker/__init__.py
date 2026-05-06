from typing import TYPE_CHECKING

from app.chunker.chunker_base import BaseChunker

if TYPE_CHECKING:  # pragma: no cover
    from app.chunker.chunker_factory import (  # noqa: F401
        chunk_document as chunk_document,
    )
    from app.chunker.chunker_factory import get_chunker as get_chunker  # noqa: F401
    from app.chunker.markdown_chunker import (  # noqa: F401
        MarkdownChunker as MarkdownChunker,
    )

__all__ = [
    "BaseChunker",
    "MarkdownChunker",
    "get_chunker",
    "chunk_document",
]


def __getattr__(name: str):
    if name == "MarkdownChunker":
        from app.chunker.markdown_chunker import MarkdownChunker

        return MarkdownChunker

    if name in {"get_chunker", "chunk_document"}:
        from app.chunker.chunker_factory import chunk_document, get_chunker

        return {
            "get_chunker": get_chunker,
            "chunk_document": chunk_document,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
