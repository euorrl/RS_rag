import json

import pytest

from app.schemas import RetrievedChunk
from evaluation.pipeline.retrieval_pipeline import (
    build_retrieval_pipeline,
    evaluate_retrieval,
    load_eval_dataset,
    run_retrieval_pipeline,
)


class FakeRecaller:
    def __init__(self):
        self.calls = []

    def recall(self, query, top_k=30, score_threshold=None):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
            }
        )
        return [
            make_chunk("a"),
            make_chunk("b"),
        ]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_n=10, score_threshold=None):
        self.calls.append(
            {
                "query": query,
                "candidates": candidates,
                "top_n": top_n,
                "score_threshold": score_threshold,
            }
        )
        return candidates[:top_n]


def make_chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=f"text {chunk_id}",
        score=1.0,
    )


def test_load_eval_dataset_reads_list(tmp_path):
    dataset_path = tmp_path / "eval_dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "question",
                    "chunks": [["a"]],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_eval_dataset(dataset_path) == [
        {
            "id": "q001",
            "question": "question",
            "chunks": [["a"]],
        }
    ]


def test_load_eval_dataset_rejects_non_list(tmp_path):
    dataset_path = tmp_path / "eval_dataset.json"
    dataset_path.write_text(json.dumps({"id": "q001"}), encoding="utf-8")

    with pytest.raises(TypeError):
        load_eval_dataset(dataset_path)


def test_build_retrieval_pipeline_uses_default_app_factories(monkeypatch):
    recaller = FakeRecaller()
    reranker = FakeReranker()

    monkeypatch.setattr("app.recaller.get_recaller", lambda: recaller)
    monkeypatch.setattr("app.reranker.get_reranker", lambda: reranker)

    retrieve = build_retrieval_pipeline()
    results = retrieve("question")

    assert [result.chunk_id for result in results] == ["a", "b"]
    assert recaller.calls == [
        {
            "query": "question",
            "top_k": 30,
            "score_threshold": 0.4,
        }
    ]
    assert reranker.calls[0]["query"] == "question"
    assert reranker.calls[0]["top_n"] == 10
    assert reranker.calls[0]["score_threshold"] == 0.5


def test_evaluate_retrieval_computes_per_sample_and_mean_metrics():
    eval_dataset = [
        {
            "id": "q001",
            "question": "question 1",
            "chunks": [["a"], ["b"]],
        },
        {
            "id": "q002",
            "question": "question 2",
            "chunks": [["c"]],
        },
    ]

    def fake_retrieve(question):
        if question == "question 1":
            return [make_chunk("x"), make_chunk("a"), make_chunk("b")]
        return [make_chunk("x"), make_chunk("y")]

    result = evaluate_retrieval(eval_dataset, fake_retrieve)

    assert result["total"] == 2
    assert result["k"] == 10
    assert result["recall_score_threshold"] == 0.4
    assert result["rerank_score_threshold"] == 0.5
    assert result["evidence_recall@10"] == [1.0, 0.0]
    assert result["mean_evidence_recall@10"] == 0.5
    assert result["mrr@10"] == [0.5, 0.0]
    assert result["mean_mrr@10"] == 0.25


def test_run_retrieval_pipeline_uses_custom_dataset_path(tmp_path):
    dataset_path = tmp_path / "custom_eval_dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "question",
                    "chunks": [["a"]],
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_retrieve(question):
        assert question == "question"
        return [make_chunk("a")]

    result = run_retrieval_pipeline(
        dataset_path=dataset_path,
        retrieve=fake_retrieve,
    )

    assert result["total"] == 1
    assert result["evidence_recall@10"] == [1.0]
    assert result["mean_evidence_recall@10"] == 1.0
    assert result["mrr@10"] == [1.0]
    assert result["mean_mrr@10"] == 1.0
