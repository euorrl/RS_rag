import pytest

from app.reranker.bge_reranker import BGEReranker
from app.reranker.reranker_factory import get_reranker, rerank
from app.schemas import RetrievedChunk

pytestmark = pytest.mark.reranker


class FakeReranker:
    """模拟 Reranker，避免 factory 测试加载真实模型。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def rerank(self, query, candidates, top_n=10, score_threshold=None):
        results = candidates
        if score_threshold is not None:
            results = [
                candidate
                for candidate in candidates
                if candidate.score >= score_threshold
            ]

        for index, candidate in enumerate(results[:top_n], start=1):
            candidate.rank = index
            candidate.rerank_method = "bge"
        return results[:top_n]


def make_candidates() -> list[RetrievedChunk]:
    """创建测试用候选 RetrievedChunk 列表。"""
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="candidate",
            score=0.5,
        )
    ]


def test_get_reranker_returns_bge_reranker(monkeypatch):
    """验证 get_reranker 默认返回 BGEReranker。"""
    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        lambda model_name: object(),
    )

    reranker = get_reranker()

    assert isinstance(reranker, BGEReranker)


def test_get_reranker_provider_case_insensitive(monkeypatch):
    """验证 get_reranker 的 provider 参数大小写不敏感。"""
    monkeypatch.setattr(
        "app.reranker.reranker_factory.BGEReranker",
        FakeReranker,
    )

    reranker = get_reranker(provider="BGE", model_name="fake-reranker")

    assert isinstance(reranker, FakeReranker)
    assert reranker.kwargs == {"model_name": "fake-reranker"}


def test_get_reranker_rejects_invalid_provider():
    """验证 get_reranker 会拒绝非法 provider 类型和不支持的 provider。"""
    with pytest.raises(TypeError):
        get_reranker(provider=123)

    with pytest.raises(ValueError):
        get_reranker(provider="unknown")


def test_rerank_convenience_entrypoint(monkeypatch):
    """验证 rerank 便捷入口会创建 reranker 并透传 score_threshold。"""
    monkeypatch.setattr(
        "app.reranker.reranker_factory.BGEReranker",
        FakeReranker,
    )
    candidates = make_candidates()

    results = rerank(
        query="query",
        candidates=candidates,
        provider="bge",
        top_n=1,
        score_threshold=0.6,
        model_name="fake-reranker",
    )

    assert results == []
