from app.embedder.embedder_base import BaseEmbedder

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
