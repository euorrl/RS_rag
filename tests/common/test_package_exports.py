import sys
from types import ModuleType

import pytest

pytestmark = pytest.mark.common


def test_schemas_init_exports_public_api():
    """验证 schemas 模块通过 __init__ 暴露统一的数据结构入口。"""
    import app.schemas as schemas

    assert schemas.__all__ == [
        "Document",
        "Chunk",
        "EmbeddedChunk",
        "RetrievedChunk",
    ]
    assert schemas.Document.__name__ == "Document"
    assert schemas.Chunk.__name__ == "Chunk"
    assert schemas.EmbeddedChunk.__name__ == "EmbeddedChunk"
    assert schemas.RetrievedChunk.__name__ == "RetrievedChunk"


def test_reader_init_exports_base_without_loading_reader_modules():
    """验证 reader 模块可直接导入轻量 BaseReader。"""
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


def test_reader_init_lazy_loads_public_api(monkeypatch):
    """验证 reader 模块通过 __getattr__ 懒加载公共入口。"""
    import app.reader as reader

    text_module = ModuleType("app.reader.text_reader")
    text_module.TextReader = type("TextReader", (), {})

    markdown_module = ModuleType("app.reader.markdown_reader")
    markdown_module.MarkdownReader = type("MarkdownReader", (), {})

    pdf_module = ModuleType("app.reader.mineru_pdf_reader")
    pdf_module.MinerUPdfReader = type("MinerUPdfReader", (), {})

    image_module = ModuleType("app.reader.mineru_image_reader")
    image_module.MinerUImageReader = type("MinerUImageReader", (), {})

    factory_module = ModuleType("app.reader.reader_factory")
    factory_module.get_reader = lambda *args, **kwargs: None
    factory_module.load_document = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, text_module.__name__, text_module)
    monkeypatch.setitem(sys.modules, markdown_module.__name__, markdown_module)
    monkeypatch.setitem(sys.modules, pdf_module.__name__, pdf_module)
    monkeypatch.setitem(sys.modules, image_module.__name__, image_module)
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)

    assert reader.__getattr__("TextReader") is text_module.TextReader
    assert reader.__getattr__("MarkdownReader") is markdown_module.MarkdownReader
    assert reader.__getattr__("MinerUPdfReader") is pdf_module.MinerUPdfReader
    assert reader.__getattr__("MinerUImageReader") is image_module.MinerUImageReader
    assert reader.__getattr__("get_reader") is factory_module.get_reader
    assert reader.__getattr__("load_document") is factory_module.load_document

    with pytest.raises(AttributeError):
        reader.__getattr__("UnknownReader")


def test_chunker_init_exports_base_without_loading_chunker_modules():
    """验证 chunker 模块可直接导入轻量 BaseChunker。"""
    import app.chunker as chunker

    assert chunker.__all__ == [
        "BaseChunker",
        "MarkdownChunker",
        "get_chunker",
        "chunk_document",
    ]
    assert chunker.BaseChunker.__name__ == "BaseChunker"


def test_chunker_init_lazy_loads_public_api(monkeypatch):
    """验证 chunker 模块通过 __getattr__ 懒加载公共入口。"""
    import app.chunker as chunker

    markdown_module = ModuleType("app.chunker.markdown_chunker")
    markdown_module.MarkdownChunker = type("MarkdownChunker", (), {})

    factory_module = ModuleType("app.chunker.chunker_factory")
    factory_module.get_chunker = lambda *args, **kwargs: None
    factory_module.chunk_document = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, markdown_module.__name__, markdown_module)
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)

    assert chunker.__getattr__("MarkdownChunker") is markdown_module.MarkdownChunker
    assert chunker.__getattr__("get_chunker") is factory_module.get_chunker
    assert chunker.__getattr__("chunk_document") is factory_module.chunk_document

    with pytest.raises(AttributeError):
        chunker.__getattr__("UnknownChunker")


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

    assert vector_store.__all__ == [
        "BaseVectorStore",
        "MilvusVectorStore",
        "get_vector_store",
        "save_embedded_chunks",
    ]
    assert vector_store.BaseVectorStore.__name__ == "BaseVectorStore"


def test_vector_store_init_lazy_loads_public_api(monkeypatch):
    """验证 vector_store 模块通过 __getattr__ 懒加载公共入口。"""
    import app.vector_store as vector_store

    milvus_module = ModuleType("app.vector_store.milvus_store")
    milvus_module.MilvusVectorStore = type("MilvusVectorStore", (), {})

    factory_module = ModuleType("app.vector_store.vector_store_factory")
    factory_module.get_vector_store = lambda *args, **kwargs: None
    factory_module.save_embedded_chunks = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, milvus_module.__name__, milvus_module)
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)

    assert (
        vector_store.__getattr__("MilvusVectorStore") is milvus_module.MilvusVectorStore
    )
    loaded_get_vector_store = vector_store.__getattr__("get_vector_store")
    assert loaded_get_vector_store is factory_module.get_vector_store
    loaded_save_embedded_chunks = vector_store.__getattr__("save_embedded_chunks")
    assert loaded_save_embedded_chunks is factory_module.save_embedded_chunks

    with pytest.raises(AttributeError):
        vector_store.__getattr__("UnknownVectorStore")
