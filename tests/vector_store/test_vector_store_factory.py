import pytest

from app.schemas import EmbeddedChunk
from app.vector_store.vector_store_factory import get_vector_store, save_embedded_chunks

pytestmark = pytest.mark.vector_store


class FakeMilvusVectorStore:
    """模拟 MilvusVectorStore，避免测试时连接真实 Milvus。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inserted_chunks = None

    def insert(self, embedded_chunks):
        self.inserted_chunks = embedded_chunks


def make_embedded_chunk(
    embedding: list[float] | None = None,
) -> EmbeddedChunk:
    """创建测试用 EmbeddedChunk。"""
    return EmbeddedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="NDVI 是一种植被指数。",
        embedding=embedding or [0.1, 0.2],
        metadata={"chunk_index": 0},
    )


def test_get_vector_store_returns_milvus_store(monkeypatch):
    """验证 get_vector_store 默认返回 MilvusVectorStore。"""
    monkeypatch.setattr(
        "app.vector_store.vector_store_factory.MilvusVectorStore",
        FakeMilvusVectorStore,
    )

    vector_store = get_vector_store(
        collection_name="demo",
        host="localhost",
        port="19530",
        dimension=2,
    )

    assert isinstance(vector_store, FakeMilvusVectorStore)
    assert vector_store.kwargs == {
        "collection_name": "demo",
        "host": "localhost",
        "port": "19530",
        "dimension": 2,
    }


def test_get_vector_store_provider_case_insensitive(monkeypatch):
    """验证 provider 大小写不敏感。"""
    monkeypatch.setattr(
        "app.vector_store.vector_store_factory.MilvusVectorStore",
        FakeMilvusVectorStore,
    )

    vector_store = get_vector_store(provider="MILVUS")

    assert isinstance(vector_store, FakeMilvusVectorStore)


def test_get_vector_store_rejects_non_string_provider():
    """验证 provider 不是字符串时抛出 TypeError。"""
    with pytest.raises(TypeError):
        get_vector_store(provider=123)


def test_get_vector_store_rejects_unsupported_provider():
    """验证 provider 不支持时抛出 ValueError。"""
    with pytest.raises(ValueError):
        get_vector_store(provider="unknown")


def test_save_embedded_chunks_infers_dimension_and_inserts(monkeypatch):
    """验证 save_embedded_chunks 会自动推断维度并写入数据。"""
    monkeypatch.setattr(
        "app.vector_store.vector_store_factory.MilvusVectorStore",
        FakeMilvusVectorStore,
    )
    chunks = [make_embedded_chunk(embedding=[0.1, 0.2, 0.3])]

    vector_store = save_embedded_chunks(
        chunks,
        provider="milvus",
        collection_name="demo",
    )

    assert isinstance(vector_store, FakeMilvusVectorStore)
    assert vector_store.kwargs == {
        "collection_name": "demo",
        "dimension": 3,
    }
    assert vector_store.inserted_chunks == chunks


def test_save_embedded_chunks_keeps_explicit_dimension(monkeypatch):
    """验证显式传入 dimension 时不会被自动推断覆盖。"""
    monkeypatch.setattr(
        "app.vector_store.vector_store_factory.MilvusVectorStore",
        FakeMilvusVectorStore,
    )
    chunks = [make_embedded_chunk(embedding=[0.1, 0.2, 0.3])]

    vector_store = save_embedded_chunks(
        chunks,
        provider="milvus",
        collection_name="demo",
        dimension=512,
    )

    assert vector_store.kwargs["dimension"] == 512
    assert vector_store.inserted_chunks == chunks


def test_save_embedded_chunks_rejects_invalid_input():
    """验证 save_embedded_chunks 会拒绝非法输入。"""
    with pytest.raises(TypeError):
        save_embedded_chunks("not-list")

    with pytest.raises(TypeError):
        save_embedded_chunks(["not-embedded-chunk"])
