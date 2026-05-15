import json

from evaluation.generation import (
    compute_claim_faithfulness,
    evaluate_answer_relevance,
    evaluate_claim_faithfulness,
    extract_claims,
    generate_questions_from_answer,
    judge_answer_relevance,
    judge_claim_support,
    normalize_answer_relevance_score,
)


def test_compute_claim_faithfulness_scores_labels():
    assert (
        compute_claim_faithfulness(["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"])
        == 0.5
    )
    assert compute_claim_faithfulness([]) == 0.0


def test_normalize_answer_relevance_score_prefers_valid_score():
    assert normalize_answer_relevance_score(1.0, "IRRELEVANT") == 1.0
    assert normalize_answer_relevance_score(0.5, None) == 0.5
    assert normalize_answer_relevance_score(None, "PARTIALLY_RELEVANT") == 0.5
    assert normalize_answer_relevance_score(0.8, "RELEVANT") == 1.0
    assert normalize_answer_relevance_score(None, None) == 0.0


def test_extract_claims_uses_answer_only_prompt_and_parses_json(monkeypatch):
    calls = []

    def fake_generate(prompt, model):
        calls.append({"prompt": prompt, "model": model})
        return json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "c1",
                        "claim": "NDVI is a vegetation index.",
                    }
                ]
            }
        )

    monkeypatch.setattr("evaluation.generation.claim_extractor.generate", fake_generate)

    claims = extract_claims("NDVI is a vegetation index.")

    assert claims == [
        {
            "claim_id": "c1",
            "claim": "NDVI is a vegetation index.",
        }
    ]
    assert calls[0]["model"] == "gpt-5.5"
    assert "context" not in calls[0]["prompt"].lower()


def test_judge_claim_support_parses_judgements(monkeypatch):
    def fake_generate(prompt, model):
        assert model == "gpt-5.5"
        assert "retrieved context" in prompt
        return json.dumps(
            {
                "judgements": [
                    {
                        "claim_id": "c1",
                        "label": "SUPPORTED",
                        "evidence": "chunk-1",
                        "reason": "The context states it.",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "evaluation.generation.faithfulness_judge.generate",
        fake_generate,
    )

    judgements = judge_claim_support(
        context="chunk-1: NDVI is a vegetation index.",
        claims=[{"claim_id": "c1", "claim": "NDVI is a vegetation index."}],
    )

    assert judgements[0]["label"] == "SUPPORTED"


def test_evaluate_claim_faithfulness_computes_score(monkeypatch):
    monkeypatch.setattr(
        "evaluation.generation.faithfulness_judge.extract_claims",
        lambda answer: [{"claim_id": "c1", "claim": answer}],
    )
    monkeypatch.setattr(
        "evaluation.generation.faithfulness_judge.judge_claim_support",
        lambda context, claims: [
            {
                "claim_id": "c1",
                "label": "PARTIALLY_SUPPORTED",
                "evidence": context,
                "reason": "Partial support.",
            }
        ],
    )

    result = evaluate_claim_faithfulness("context", "answer claim")

    assert result["claims"] == [{"claim_id": "c1", "claim": "answer claim"}]
    assert result["claim_faithfulness"] == 0.5


def test_evaluate_claim_faithfulness_returns_zero_for_no_claims(monkeypatch):
    monkeypatch.setattr(
        "evaluation.generation.faithfulness_judge.extract_claims",
        lambda answer: [],
    )

    result = evaluate_claim_faithfulness("context", "no factual claims")

    assert result == {
        "claims": [],
        "judgements": [],
        "claim_faithfulness": 0.0,
    }


def test_generate_questions_from_answer_parses_json(monkeypatch):
    def fake_generate(prompt, model):
        assert model == "gpt-5.5"
        assert "answer:" in prompt.lower()
        return json.dumps({"generated_questions": ["What is NDVI?"]})

    monkeypatch.setattr(
        "evaluation.generation.question_generator.generate",
        fake_generate,
    )

    assert generate_questions_from_answer("NDVI is a vegetation index.") == [
        "What is NDVI?"
    ]


def test_judge_answer_relevance_normalizes_score(monkeypatch):
    def fake_generate(prompt, model):
        assert model == "gpt-5.5"
        assert "generated_questions" in prompt
        return json.dumps(
            {
                "score": 0.8,
                "label": "RELEVANT",
                "reason": "Same meaning.",
            }
        )

    monkeypatch.setattr(
        "evaluation.generation.answer_relevance_judge.generate",
        fake_generate,
    )

    result = judge_answer_relevance("What is NDVI?", ["What is NDVI?"])

    assert result == {
        "score": 1.0,
        "label": "RELEVANT",
        "reason": "Same meaning.",
    }


def test_evaluate_answer_relevance_keeps_flow_separate(monkeypatch):
    monkeypatch.setattr(
        "evaluation.generation.answer_relevance_judge.generate_questions_from_answer",
        lambda answer: [f"What does this answer say about {answer}?"],
    )
    monkeypatch.setattr(
        "evaluation.generation.answer_relevance_judge.judge_answer_relevance",
        lambda question, generated_questions: {
            "score": 0.5,
            "label": "PARTIALLY_RELEVANT",
            "reason": question + " partially matches.",
        },
    )

    result = evaluate_answer_relevance("What is NDVI?", "NDVI")

    assert result["generated_questions"] == ["What does this answer say about NDVI?"]
    assert result["score"] == 0.5
    assert result["label"] == "PARTIALLY_RELEVANT"
