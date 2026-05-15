"""RAG 生成阶段的事实断言抽取。"""

from __future__ import annotations

import json

from app.generator import generate
from evaluation.generation.prompts import build_claim_extraction_prompt


def extract_claims(answer: str) -> list[dict]:
    """从生成答案中抽取原子事实断言。

    Args:
        answer: 生成答案文本。

    Returns:
        抽取出的事实断言对象列表。
    """
    prompt = build_claim_extraction_prompt(answer)
    response = generate(prompt, model="gpt-5.5")
    data = json.loads(response)
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        return []

    return claims
