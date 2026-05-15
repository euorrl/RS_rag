import json

import pytest

from app.schemas import RetrievedChunk
from evaluation.pipeline.generation_pipeline import (
    build_context_from_chunks,
    build_generation_pipeline,
    build_strict_rag_prompt,
    evaluate_generation,
    load_eval_dataset,
    run_generation_pipeline,
)


def make_chunk(chunk_id: str, text: str = "chunk text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
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


def test_build_context_from_chunks_uses_chunk_id_and_text():
    context = build_context_from_chunks(
        [
            make_chunk("a", "text a"),
            {"chunk_id": "b", "text": "text b"},
        ]
    )

    assert context == "[chunk_id: a]\ntext a\n\n[chunk_id: b]\ntext b"


def test_build_strict_rag_prompt_disallows_external_knowledge():
    prompt = build_strict_rag_prompt(
        question="什么是 NDVI?",
        context="[chunk_id: a]\nNDVI 是归一化植被指数。",
    )

    assert "请只根据给定 context 回答问题" in prompt
    assert "不要使用外部知识" in prompt
    assert "给定资料不足" in prompt
    assert "什么是 NDVI?" in prompt
    assert "NDVI 是归一化植被指数。" in prompt


def test_build_generation_pipeline_uses_default_app_components(monkeypatch):
    class FakeRecaller:
        def __init__(self):
            self.calls = []

        def recall(self, query):
            self.calls.append(query)
            return [make_chunk("a", "recalled text")]

    class FakeReranker:
        def __init__(self):
            self.calls = []

        def rerank(self, query, candidates):
            self.calls.append(
                {
                    "query": query,
                    "candidates": candidates,
                }
            )
            return [make_chunk("b", "reranked text")]

    recaller = FakeRecaller()
    reranker = FakeReranker()
    generate_calls = []

    def fake_generate(prompt, model):
        generate_calls.append({"prompt": prompt, "model": model})
        return "strict answer"

    monkeypatch.setattr("app.recaller.get_recaller", lambda: recaller)
    monkeypatch.setattr("app.reranker.get_reranker", lambda: reranker)
    monkeypatch.setattr("app.generator.generate", fake_generate)

    generate_answer = build_generation_pipeline()
    result = generate_answer("question")

    assert recaller.calls == ["question"]
    assert reranker.calls[0]["query"] == "question"
    assert [chunk.chunk_id for chunk in reranker.calls[0]["candidates"]] == ["a"]
    assert generate_calls[0]["model"] == "gpt-5.5"
    assert "不要使用外部知识" in generate_calls[0]["prompt"]
    assert result == {
        "answer": "strict answer",
        "context": "[chunk_id: b]\nreranked text",
        "retrieved_chunks": [make_chunk("b", "reranked text")],
    }


def test_evaluate_generation_computes_items_and_means(monkeypatch):
    eval_dataset = [
        {
            "id": "q001",
            "question": "question 1",
            "chunks": [["gold-a"]],
        },
        {
            "id": "q002",
            "question": "question 2",
            "chunks": [["gold-b"]],
        },
    ]

    def fake_generate_answer(question):
        return {
            "answer": f"answer for {question}",
            "context": "" if question == "question 1" else f"context for {question}",
            "retrieved_chunks": [
                make_chunk(f"retrieved-{question[-1]}"),
                {"chunk_id": f"dict-{question[-1]}", "text": "dict text"},
            ],
        }

    def fake_evaluate_claim_faithfulness(context, answer):
        score = 1.0 if "retrieved-1" in context else 0.5
        return {
            "claims": [],
            "judgements": [],
            "claim_faithfulness": score,
        }

    def fake_evaluate_answer_relevance(question, answer):
        score = 1.0 if question == "question 1" else 0.0
        return {
            "generated_questions": [],
            "score": score,
            "label": "RELEVANT",
            "reason": answer,
        }

    monkeypatch.setattr(
        "evaluation.pipeline.generation_pipeline.evaluate_claim_faithfulness",
        fake_evaluate_claim_faithfulness,
    )
    monkeypatch.setattr(
        "evaluation.pipeline.generation_pipeline.evaluate_answer_relevance",
        fake_evaluate_answer_relevance,
    )

    result = evaluate_generation(eval_dataset, fake_generate_answer)

    assert result["total"] == 2
    assert result["mean_claim_faithfulness"] == 0.75
    assert result["mean_answer_relevance"] == 0.5
    assert result["items"][0]["golden_chunks"] == [["gold-a"]]
    assert result["items"][0]["retrieved_ids"] == ["retrieved-1", "dict-1"]
    assert result["items"][1]["retrieved_ids"] == ["retrieved-2", "dict-2"]


def test_evaluate_generation_returns_zero_means_for_empty_dataset():
    result = evaluate_generation([], lambda question: {})

    assert result == {
        "total": 0,
        "mean_claim_faithfulness": 0.0,
        "mean_answer_relevance": 0.0,
        "items": [],
    }


def test_run_generation_pipeline_uses_custom_dataset_path(tmp_path, monkeypatch):
    dataset_path = tmp_path / "custom_eval_dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "question",
                    "chunks": [["gold-a"]],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "evaluation.pipeline.generation_pipeline.evaluate_claim_faithfulness",
        lambda context, answer: {
            "claims": [],
            "judgements": [],
            "claim_faithfulness": 1.0,
        },
    )
    monkeypatch.setattr(
        "evaluation.pipeline.generation_pipeline.evaluate_answer_relevance",
        lambda question, answer: {
            "generated_questions": [],
            "score": 1.0,
            "label": "RELEVANT",
            "reason": "",
        },
    )

    def fake_generate_answer(question):
        assert question == "question"
        return {
            "answer": "answer",
            "context": "context",
            "retrieved_chunks": [make_chunk("retrieved-a")],
        }

    result = run_generation_pipeline(
        dataset_path=dataset_path,
        generate_answer=fake_generate_answer,
    )

    assert result["total"] == 1
    assert result["mean_claim_faithfulness"] == 1.0
    assert result["mean_answer_relevance"] == 1.0
    assert result["items"][0]["retrieved_ids"] == ["retrieved-a"]


def test_run_generation_pipeline_builds_default_pipeline(tmp_path, monkeypatch):
    dataset_path = tmp_path / "custom_eval_dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "question",
                    "chunks": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_build_generation_pipeline():
        return lambda question: {
            "answer": f"answer for {question}",
            "context": "context",
            "retrieved_chunks": [],
        }

    monkeypatch.setattr(
        "evaluation.pipeline.generation_pipeline.build_generation_pipeline",
        fake_build_generation_pipeline,
    )
    monkeypatch.setattr(
        "evaluation.pipeline.generation_pipeline.evaluate_claim_faithfulness",
        lambda context, answer: {
            "claims": [],
            "judgements": [],
            "claim_faithfulness": 1.0,
        },
    )
    monkeypatch.setattr(
        "evaluation.pipeline.generation_pipeline.evaluate_answer_relevance",
        lambda question, answer: {
            "generated_questions": [],
            "score": 1.0,
            "label": "RELEVANT",
            "reason": "",
        },
    )

    result = run_generation_pipeline(dataset_path=dataset_path)

    assert result["total"] == 1
    assert result["items"][0]["answer"] == "answer for question"
