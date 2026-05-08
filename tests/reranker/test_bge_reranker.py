import sys
from types import ModuleType

import pytest

from app.reranker import bge_reranker
from app.reranker.bge_reranker import BGEReranker
from app.schemas import RetrievedChunk

pytestmark = pytest.mark.reranker


class FakeScores:
    """模拟 numpy.ndarray 的 tolist 行为。"""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeCrossEncoder:
    """模拟 CrossEncoder，避免测试时加载真实 reranker 模型。"""

    scores = [0.2, 0.9, 0.5]

    def __init__(self, model_name):
        self.model_name = model_name
        self.predict_calls = []

    def predict(self, pairs):
        self.predict_calls.append(pairs)
        return FakeScores(self.scores)


class BadCountCrossEncoder(FakeCrossEncoder):
    """模拟返回分数数量错误的 CrossEncoder。"""

    def predict(self, pairs):
        return [0.1]


class BadScoreCrossEncoder(FakeCrossEncoder):
    """模拟返回非数字分数的 CrossEncoder。"""

    def predict(self, pairs):
        return [0.1, "bad", 0.3]


def make_candidates() -> list[RetrievedChunk]:
    """创建测试用候选 RetrievedChunk 列表。"""
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="candidate one",
            score=0.1,
            score_details={"vector_score": 0.1},
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            text="candidate two",
            score=0.2,
            score_details={"vector_score": 0.2},
        ),
        RetrievedChunk(
            chunk_id="chunk-3",
            document_id="doc-1",
            text="candidate three",
            score=0.3,
            score_details={"vector_score": 0.3},
        ),
    ]


def test_bge_reranker_initializes_model(monkeypatch):
    """验证 BGEReranker 初始化时会加载指定的 CrossEncoder 模型。"""
    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        FakeCrossEncoder,
    )

    reranker = BGEReranker(model_name="fake-reranker")

    assert reranker.model_name == "fake-reranker"
    assert reranker.model.model_name == "fake-reranker"


def test_get_cross_encoder_lazy_loads_sentence_transformers(monkeypatch):
    """验证 _get_cross_encoder 会在需要时懒加载 sentence_transformers。"""
    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder

    monkeypatch.setattr("app.reranker.bge_reranker.CrossEncoder", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    cross_encoder = bge_reranker._get_cross_encoder()

    assert cross_encoder is FakeCrossEncoder


def test_bge_reranker_wraps_model_load_failure(monkeypatch):
    """验证 BGEReranker 会包装模型加载失败异常。"""

    def fail_cross_encoder(model_name):
        raise OSError("load failed")

    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        fail_cross_encoder,
    )

    with pytest.raises(RuntimeError):
        BGEReranker(model_name="bad-model")


def test_bge_reranker_reranks_candidates_by_score(monkeypatch):
    """验证 BGEReranker 会按 rerank 分数降序排序并截取 top_n。"""
    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        FakeCrossEncoder,
    )
    candidates = make_candidates()
    reranker = BGEReranker(model_name="fake-reranker")

    results = reranker.rerank("query", candidates, top_n=2)

    assert reranker.model.predict_calls == [
        [
            ["query", "candidate one"],
            ["query", "candidate two"],
            ["query", "candidate three"],
        ]
    ]
    assert [result.chunk_id for result in results] == ["chunk-2", "chunk-3"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.score for result in results] == [0.9, 0.5]
    assert results[0].recall_method == "vector"
    assert results[0].rerank_method == "bge"
    assert results[0].score_details["vector_score"] == 0.2
    assert results[0].score_details["rerank_score"] == 0.9
    assert "reranker_model" not in results[0].score_details


def test_bge_reranker_returns_empty_list_for_empty_candidates(monkeypatch):
    """验证空候选列表会直接返回空列表且不调用模型。"""
    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        FakeCrossEncoder,
    )
    reranker = BGEReranker(model_name="fake-reranker")

    assert reranker.rerank("query", []) == []
    assert reranker.model.predict_calls == []


def test_bge_reranker_filters_results_by_score_threshold(monkeypatch):
    """验证 BGEReranker 会过滤低于 score_threshold 的重排结果。"""
    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        FakeCrossEncoder,
    )
    candidates = make_candidates()
    reranker = BGEReranker(model_name="fake-reranker")

    results = reranker.rerank("query", candidates, score_threshold=0.6)

    assert [result.chunk_id for result in results] == ["chunk-2"]
    assert results[0].score == 0.9
    assert results[0].rank == 1


def test_bge_reranker_does_not_filter_scores_when_threshold_is_none(monkeypatch):
    """验证 score_threshold 为 None 时不会按分数过滤重排结果。"""
    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        FakeCrossEncoder,
    )
    candidates = make_candidates()
    reranker = BGEReranker(model_name="fake-reranker")

    results = reranker.rerank("query", candidates, score_threshold=None)

    assert [result.chunk_id for result in results] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]


def test_bge_reranker_rejects_invalid_input(monkeypatch):
    """验证 BGEReranker 会拒绝非法 query、candidates、top_n 和阈值。"""
    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        FakeCrossEncoder,
    )
    reranker = BGEReranker(model_name="fake-reranker")

    with pytest.raises(TypeError):
        reranker.rerank(123, make_candidates())

    with pytest.raises(ValueError):
        reranker.rerank("   ", make_candidates())

    with pytest.raises(TypeError):
        reranker.rerank("query", "not-list")

    with pytest.raises(TypeError):
        reranker.rerank("query", ["not-retrieved-chunk"])

    with pytest.raises(ValueError):
        reranker.rerank("query", make_candidates(), top_n=0)

    with pytest.raises(TypeError):
        reranker.rerank("query", make_candidates(), score_threshold="0.6")


def test_bge_reranker_raises_when_score_count_mismatches(monkeypatch):
    """验证 rerank 分数数量与候选数量不一致时抛出 RuntimeError。"""
    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        BadCountCrossEncoder,
    )
    reranker = BGEReranker(model_name="fake-reranker")

    with pytest.raises(RuntimeError):
        reranker.rerank("query", make_candidates())


def test_bge_reranker_rejects_non_numeric_scores(monkeypatch):
    """验证 rerank 分数包含非数字值时抛出 TypeError。"""
    monkeypatch.setattr(
        "app.reranker.bge_reranker.CrossEncoder",
        BadScoreCrossEncoder,
    )
    reranker = BGEReranker(model_name="fake-reranker")

    with pytest.raises(TypeError):
        reranker.rerank("query", make_candidates())
