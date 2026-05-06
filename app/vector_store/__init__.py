from typing import TYPE_CHECKING

from app.vector_store.vector_store_base import BaseVectorStore

if TYPE_CHECKING:  # pragma: no cover
    from app.vector_store.milvus_store import (  # noqa: F401
        MilvusVectorStore as MilvusVectorStore,
    )
    from app.vector_store.vector_store_factory import (  # noqa: F401
        get_vector_store as get_vector_store,
    )
    from app.vector_store.vector_store_factory import (  # noqa: F401
        save_embedded_chunks as save_embedded_chunks,
    )

__all__ = [
    "BaseVectorStore",
    "MilvusVectorStore",
    "get_vector_store",
    "save_embedded_chunks",
]


def __getattr__(name: str):
    if name == "MilvusVectorStore":
        from app.vector_store.milvus_store import MilvusVectorStore

        return MilvusVectorStore

    if name == "get_vector_store":
        from app.vector_store.vector_store_factory import get_vector_store

        return get_vector_store

    if name == "save_embedded_chunks":
        from app.vector_store.vector_store_factory import save_embedded_chunks

        return save_embedded_chunks

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
