import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.common


def test_schemas_init_exports_public_api():
    """验证 schemas 模块通过 __init__ 暴露统一的数据结构入口。"""
    import app.schemas as schemas

    assert schemas.__all__ == ["Document", "Chunk", "EmbeddedChunk"]
    assert schemas.Document.__name__ == "Document"
    assert schemas.Chunk.__name__ == "Chunk"
    assert schemas.EmbeddedChunk.__name__ == "EmbeddedChunk"


def test_reader_init_exports_public_api():
    """验证 reader 模块通过 __init__ 暴露统一的 Reader 入口。"""
    import app.reader as reader

    assert reader.__all__ == [
        "BaseReader",
        "TextReader",
        "MarkdownReader",
        "MinerUPdfReader",
        "MinerUImageReader",
        "get_reader",
        "load_document",
    ]
    assert reader.BaseReader.__name__ == "BaseReader"
    assert reader.TextReader.__name__ == "TextReader"
    assert reader.MarkdownReader.__name__ == "MarkdownReader"
    assert reader.MinerUPdfReader.__name__ == "MinerUPdfReader"
    assert reader.MinerUImageReader.__name__ == "MinerUImageReader"
    assert callable(reader.get_reader)
    assert callable(reader.load_document)


def test_chunker_init_exports_public_api():
    """验证 chunker 模块通过 __init__ 暴露统一的 Chunker 入口。"""
    import app.chunker as chunker

    assert chunker.__all__ == [
        "BaseChunker",
        "MarkdownChunker",
        "get_chunker",
        "chunk_document",
    ]
    assert chunker.BaseChunker.__name__ == "BaseChunker"
    assert chunker.MarkdownChunker.__name__ == "MarkdownChunker"
    assert callable(chunker.get_chunker)
    assert callable(chunker.chunk_document)


def test_embedder_init_exports_base_without_loading_heavy_modules():
    """验证 embedder 模块可直接导入轻量 BaseEmbedder。"""
    import app.embedder as embedder

    assert embedder.__all__ == [
        "BaseEmbedder",
        "BGEEmbedder",
        "SentenceTransformerEmbedder",
        "get_embedder",
        "embed_chunks",
    ]
    assert embedder.BaseEmbedder.__name__ == "BaseEmbedder"


def test_embedder_init_lazy_loads_public_api(monkeypatch):
    """验证 embedder 模块通过 __getattr__ 懒加载重依赖入口。"""
    import app.embedder as embedder

    bge_module = ModuleType("app.embedder.bge_embedder")
    bge_module.BGEEmbedder = type("BGEEmbedder", (), {})

    minilm_module = ModuleType("app.embedder.minilm_embedder")
    minilm_module.SentenceTransformerEmbedder = type(
        "SentenceTransformerEmbedder",
        (),
        {},
    )

    factory_module = ModuleType("app.embedder.embedder_factory")
    factory_module.get_embedder = lambda *args, **kwargs: None
    factory_module.embed_chunks = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, bge_module.__name__, bge_module)
    monkeypatch.setitem(sys.modules, minilm_module.__name__, minilm_module)
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)

    assert embedder.__getattr__("BGEEmbedder") is bge_module.BGEEmbedder
    assert (
        embedder.__getattr__("SentenceTransformerEmbedder")
        is minilm_module.SentenceTransformerEmbedder
    )
    assert embedder.__getattr__("get_embedder") is factory_module.get_embedder
    assert embedder.__getattr__("embed_chunks") is factory_module.embed_chunks

    with pytest.raises(AttributeError):
        embedder.__getattr__("UnknownEmbedder")


def test_vector_store_init_exports_base_without_loading_milvus():
    """验证 vector_store 模块可直接导入轻量 BaseVectorStore。"""
    import app.vector_store as vector_store

    assert vector_store.__all__ == ["BaseVectorStore", "MilvusVectorStore"]
    assert vector_store.BaseVectorStore.__name__ == "BaseVectorStore"


def test_vector_store_init_lazy_loads_milvus_store(monkeypatch):
    """验证 vector_store 模块通过 __getattr__ 懒加载 Milvus 入口。"""
    import app.vector_store as vector_store

    milvus_module = ModuleType("app.vector_store.milvus_store")
    milvus_module.MilvusVectorStore = type("MilvusVectorStore", (), {})

    monkeypatch.setitem(sys.modules, milvus_module.__name__, milvus_module)

    assert (
        vector_store.__getattr__("MilvusVectorStore") is milvus_module.MilvusVectorStore
    )

    with pytest.raises(AttributeError):
        vector_store.__getattr__("UnknownVectorStore")
