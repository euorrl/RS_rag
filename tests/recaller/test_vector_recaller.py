import pytest

from app.embedder import BaseEmbedder
from app.recaller import VectorRecaller
from app.schemas import EmbeddedChunk, RetrievedChunk
from app.vector_store import BaseVectorStore

pytestmark = pytest.mark.recaller


class FakeEmbedder(BaseEmbedder):
    """模拟 Embedder，并记录 query embedding 调用。"""

    def __init__(self):
        self.queries = []

    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, query):
        self.queries.append(query)
        return [0.1, 0.2]


class FakeVectorStore(BaseVectorStore):
    """模拟 VectorStore，并记录 search 调用参数。"""

    def __init__(self, results=None):
        self.results = results
        self.search_calls = []

    def create_collection(self):
        return None

    def insert(self, embedded_chunks: list[EmbeddedChunk]):
        return None

    def search(self, query_vector, top_k=5):
        self.search_calls.append(
            {
                "query_vector": query_vector,
                "top_k": top_k,
            }
        )
        if self.results is not None:
            return self.results

        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="NDVI is a vegetation index.",
                score=0.98,
                metadata={"chunk_index": 0},
                milvus_id=10,
            )
        ]

    def drop_collection(self):
        return None


def test_vector_recaller_embeds_query_and_searches_vector_store():
    """验证 VectorRecaller 会先向量化 query，再调用向量库 search。"""
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    recaller = VectorRecaller(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = recaller.recall("what is NDVI?", top_k=3)

    assert embedder.queries == ["what is NDVI?"]
    assert vector_store.search_calls == [
        {
            "query_vector": [0.1, 0.2],
            "top_k": 3,
        }
    ]
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].rank == 1
    assert results[0].recall_method == "vector"
    assert results[0].rerank_method is None
    assert results[0].score_details == {"vector_score": 0.98}


def test_vector_recaller_preserves_existing_score_details():
    """验证已有 score_details 不会被 VectorRecaller 覆盖。"""
    result = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="text",
        score=0.8,
        score_details={"vector_score": 0.7},
    )
    recaller = VectorRecaller(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(results=[result]),
    )

    results = recaller.recall("query")

    assert results[0].score_details == {"vector_score": 0.7}


def test_vector_recaller_uses_default_top_k_and_score_threshold():
    """验证 VectorRecaller 默认召回 top_k=30，并过滤低于 0.4 的结果。"""
    results = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="high score",
            score=0.41,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            text="low score",
            score=0.39,
        ),
    ]
    vector_store = FakeVectorStore(results=results)
    recaller = VectorRecaller(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
    )

    recalled_results = recaller.recall("query")

    assert vector_store.search_calls[0]["top_k"] == 30
    assert [result.chunk_id for result in recalled_results] == ["chunk-1"]


def test_vector_recaller_filters_results_by_score_threshold():
    """验证 VectorRecaller 会过滤低于 score_threshold 的召回结果。"""
    results = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="high score",
            score=0.82,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            text="low score",
            score=0.61,
        ),
    ]
    recaller = VectorRecaller(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(results=results),
    )

    filtered_results = recaller.recall("query", score_threshold=0.8)

    assert len(filtered_results) == 1
    assert filtered_results[0].chunk_id == "chunk-1"
    assert filtered_results[0].rank == 1
    assert filtered_results[0].score_details == {"vector_score": 0.82}


def test_vector_recaller_returns_empty_list_when_all_scores_are_too_low():
    """验证所有召回结果低于 score_threshold 时返回空列表。"""
    results = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="low score",
            score=0.45,
        )
    ]
    recaller = VectorRecaller(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(results=results),
    )

    assert recaller.recall("query", score_threshold=0.8) == []


def test_vector_recaller_does_not_filter_scores_when_threshold_is_none():
    """验证 score_threshold 为 None 时不会按分数过滤召回结果。"""
    results = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="low score",
            score=0.1,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            text="high score",
            score=0.9,
        ),
    ]
    recaller = VectorRecaller(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(results=results),
    )

    recalled_results = recaller.recall("query", score_threshold=None)

    assert [result.chunk_id for result in recalled_results] == [
        "chunk-1",
        "chunk-2",
    ]
    assert [result.rank for result in recalled_results] == [1, 2]


def test_vector_recaller_rejects_invalid_dependencies():
    """验证 VectorRecaller 会拒绝非法 embedder 和 vector_store 依赖。"""
    with pytest.raises(TypeError):
        VectorRecaller(
            embedder="not-embedder",
            vector_store=FakeVectorStore(),
        )

    with pytest.raises(TypeError):
        VectorRecaller(
            embedder=FakeEmbedder(),
            vector_store="not-vector-store",
        )


def test_vector_recaller_rejects_invalid_query():
    """验证 VectorRecaller 会拒绝非法 query、top_k 和 score_threshold 参数。"""
    recaller = VectorRecaller(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(TypeError):
        recaller.recall(123)

    with pytest.raises(ValueError):
        recaller.recall("   ")

    with pytest.raises(ValueError):
        recaller.recall("query", top_k=0)

    with pytest.raises(TypeError):
        recaller.recall("query", score_threshold="0.8")


def test_vector_recaller_rejects_invalid_search_results():
    """验证 VectorRecaller 会校验 vector_store.search 的返回类型。"""
    recaller = VectorRecaller(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(results="not-list"),
    )

    with pytest.raises(TypeError):
        recaller.recall("query")

    recaller = VectorRecaller(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(results=["not-result"]),
    )

    with pytest.raises(TypeError):
        recaller.recall("query")
