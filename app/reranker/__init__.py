from typing import TYPE_CHECKING

from app.reranker.reranker_base import BaseReranker

if TYPE_CHECKING:  # pragma: no cover
    from app.reranker.bge_reranker import BGEReranker as BGEReranker
    from app.reranker.reranker_factory import get_reranker as get_reranker
    from app.reranker.reranker_factory import rerank as rerank

__all__ = [
    "BaseReranker",
    "BGEReranker",
    "get_reranker",
    "rerank",
]


def __getattr__(name: str):
    if name == "BGEReranker":
        from app.reranker.bge_reranker import BGEReranker

        return BGEReranker

    if name in {"get_reranker", "rerank"}:
        from app.reranker.reranker_factory import get_reranker, rerank

        return {
            "get_reranker": get_reranker,
            "rerank": rerank,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
