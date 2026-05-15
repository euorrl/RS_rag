"""用于答案相关性评估的答案反推问题生成。"""

from __future__ import annotations

import json

from app.generator import generate
from evaluation.generation.prompts import build_answer_to_questions_prompt


def generate_questions_from_answer(answer: str) -> list[str]:
    """生成该答案可能正在回答的问题。

    Args:
        answer: 生成答案文本。

    Returns:
        仅根据答案反推出的问题列表。
    """
    prompt = build_answer_to_questions_prompt(answer)
    response = generate(prompt, model="gpt-5.5")
    data = json.loads(response)
    generated_questions = data.get("generated_questions", [])
    if not isinstance(generated_questions, list):
        return []

    return [
        generated_question
        for generated_question in generated_questions
        if isinstance(generated_question, str)
    ]
