from typing import TYPE_CHECKING

from app.recaller.recaller_base import BaseRecaller

if TYPE_CHECKING:  # pragma: no cover
    from app.recaller.recaller_factory import get_recaller as get_recaller
    from app.recaller.recaller_factory import recall as recall
    from app.recaller.vector_recaller import VectorRecaller as VectorRecaller

__all__ = [
    "BaseRecaller",
    "VectorRecaller",
    "get_recaller",
    "recall",
]


def __getattr__(name: str):
    if name == "VectorRecaller":
        from app.recaller.vector_recaller import VectorRecaller

        return VectorRecaller

    if name in {"get_recaller", "recall"}:
        from app.recaller.recaller_factory import get_recaller, recall

        return {
            "get_recaller": get_recaller,
            "recall": recall,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
