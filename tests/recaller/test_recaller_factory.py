import pytest

from app.embedder import BaseEmbedder
from app.recaller import VectorRecaller
from app.recaller.recaller_factory import get_recaller, recall
from app.schemas import EmbeddedChunk, RetrievedChunk
from app.vector_store import BaseVectorStore

pytestmark = pytest.mark.recaller


class FakeEmbedder(BaseEmbedder):
    """模拟 Embedder，避免测试时加载真实 embedding 模型。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_texts(self, texts):
        return [[1.0, 2.0] for _ in texts]

    def embed_query(self, query):
        return [1.0, 2.0]


class FakeVectorStore(BaseVectorStore):
    """模拟 VectorStore，避免测试时连接真实 Milvus。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_collection(self):
        return None

    def insert(self, embedded_chunks: list[EmbeddedChunk]):
        return None

    def search(self, query_vector, top_k=5):
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="text",
                score=0.9,
            )
        ]

    def drop_collection(self):
        return None


def test_get_recaller_returns_vector_recaller_with_existing_dependencies():
    """验证 get_recaller 可使用外部传入的 embedder 和 vector_store。"""
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    recaller = get_recaller(
        provider="vector",
        embedder=embedder,
        vector_store=vector_store,
    )

    assert isinstance(recaller, VectorRecaller)
    assert recaller.embedder is embedder
    assert recaller.vector_store is vector_store


def test_get_recaller_provider_case_insensitive():
    """验证 get_recaller 的 provider 参数大小写不敏感。"""
    recaller = get_recaller(
        provider="VECTOR",
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    assert isinstance(recaller, VectorRecaller)


def test_get_recaller_creates_missing_dependencies(monkeypatch):
    """验证 get_recaller 会在未传入依赖时自动创建 embedder 和 vector_store。"""
    calls = []

    def fake_get_embedder(provider="bge", **kwargs):
        calls.append(("embedder", provider, kwargs))
        return FakeEmbedder(**kwargs)

    def fake_get_vector_store(provider="milvus", **kwargs):
        calls.append(("vector_store", provider, kwargs))
        return FakeVectorStore(**kwargs)

    monkeypatch.setattr(
        "app.embedder.get_embedder",
        fake_get_embedder,
        raising=False,
    )
    monkeypatch.setattr(
        "app.vector_store.get_vector_store",
        fake_get_vector_store,
        raising=False,
    )

    recaller = get_recaller(
        provider="vector",
        embedder_provider="minilm",
        vector_store_provider="milvus",
        embedder_kwargs={"model_name": "fake-model"},
        vector_store_kwargs={
            "collection_name": "test_chunks",
            "dimension": 2,
        },
    )

    assert isinstance(recaller, VectorRecaller)
    assert isinstance(recaller.embedder, FakeEmbedder)
    assert isinstance(recaller.vector_store, FakeVectorStore)
    assert recaller.embedder.kwargs == {"model_name": "fake-model"}
    assert recaller.vector_store.kwargs == {
        "collection_name": "test_chunks",
        "dimension": 2,
    }
    assert calls == [
        ("embedder", "minilm", {"model_name": "fake-model"}),
        (
            "vector_store",
            "milvus",
            {
                "collection_name": "test_chunks",
                "dimension": 2,
            },
        ),
    ]


def test_get_recaller_rejects_invalid_provider():
    """验证 get_recaller 会拒绝非法 provider 类型和不支持的 provider。"""
    with pytest.raises(TypeError):
        get_recaller(provider=123)

    with pytest.raises(ValueError):
        get_recaller(provider="bm25")


def test_recall_convenience_entrypoint():
    """验证 recall 便捷入口会创建 recaller 并透传 score_threshold。"""
    results = recall(
        "query",
        score_threshold=0.95,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    assert results == []
