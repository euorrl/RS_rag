"""RAG 生成阶段的答案忠实度评估。"""

from __future__ import annotations

import json
from typing import Any

from app.generator import generate
from evaluation.generation.claim_extractor import extract_claims
from evaluation.generation.metrics import compute_claim_faithfulness
from evaluation.generation.prompts import build_claim_support_prompt


def judge_claim_support(
    context: str,
    claims: list[dict],
) -> list[dict]:
    """判断抽取出的事实断言是否被检索上下文支持。

    Args:
        context: RAG 生成时使用的检索上下文。
        claims: 从生成答案中抽取出的事实断言。

    Returns:
        事实断言支持性判断对象列表。
    """
    prompt = build_claim_support_prompt(context, claims)
    response = generate(prompt, model="gpt-5.5")
    data = json.loads(response)
    judgements = data.get("judgements", [])
    if not isinstance(judgements, list):
        return []

    return judgements


def evaluate_claim_faithfulness(
    context: str,
    answer: str,
) -> dict[str, Any]:
    """评估生成答案的事实断言忠实度。

    Args:
        context: RAG 生成时使用的检索上下文。
        answer: 生成答案文本。

    Returns:
        包含断言、支持性判断和代码计算分数的完整评估结果。
    """
    claims = extract_claims(answer)
    if not claims:
        return {
            "claims": [],
            "judgements": [],
            "claim_faithfulness": 0.0,
        }

    judgements = judge_claim_support(context, claims)
    labels = [
        judgement.get("label", "")
        for judgement in judgements
        if isinstance(judgement.get("label"), str)
    ]
    claim_faithfulness = compute_claim_faithfulness(labels)

    return {
        "claims": claims,
        "judgements": judgements,
        "claim_faithfulness": claim_faithfulness,
    }
