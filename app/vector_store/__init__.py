from app.vector_store.vector_store_base import BaseVectorStore

__all__ = [
    "BaseVectorStore",
    "MilvusVectorStore",
]


def __getattr__(name: str):
    if name == "MilvusVectorStore":
        from app.vector_store.milvus_store import MilvusVectorStore

        return MilvusVectorStore

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
