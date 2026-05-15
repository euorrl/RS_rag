"""RAG 生成阶段的答案相关性评估。"""

from __future__ import annotations

import json
from typing import Any

from app.generator import generate
from evaluation.generation.metrics import normalize_answer_relevance_score
from evaluation.generation.prompts import build_answer_relevance_judgement_prompt
from evaluation.generation.question_generator import generate_questions_from_answer


def judge_answer_relevance(
    question: str,
    generated_questions: list[str],
) -> dict[str, Any]:
    """判断反推问题是否与原始问题匹配。

    Args:
        question: 原始用户问题。
        generated_questions: 仅根据生成答案反推出的问题列表。

    Returns:
        包含规范化分数、标签和原因的评估结果。
    """
    prompt = build_answer_relevance_judgement_prompt(
        question,
        generated_questions,
    )
    response = generate(prompt, model="gpt-5.5")
    data = json.loads(response)
    label = data.get("label")
    score = normalize_answer_relevance_score(data.get("score"), label)

    return {
        "score": score,
        "label": label,
        "reason": data.get("reason", ""),
    }


def evaluate_answer_relevance(
    question: str,
    answer: str,
) -> dict[str, Any]:
    """评估生成答案是否回答了原始问题。

    Args:
        question: 原始用户问题。
        answer: 生成答案文本。

    Returns:
        包含反推问题和相关性判断的完整评估结果。
    """
    generated_questions = generate_questions_from_answer(answer)
    judgement = judge_answer_relevance(question, generated_questions)

    return {
        "generated_questions": generated_questions,
        "score": judgement["score"],
        "label": judgement["label"],
        "reason": judgement["reason"],
    }
