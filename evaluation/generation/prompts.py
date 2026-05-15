"""RAG 生成阶段评估的 prompt 构造函数。"""

from __future__ import annotations

import json


def build_claim_extraction_prompt(answer: str) -> str:
    """构造从答案中抽取原子事实断言的 prompt。

    Args:
        answer: 生成答案文本。

    Returns:
        要求 LLM 仅以 JSON 返回事实断言的 prompt 文本。
    """
    return f"""你是一个严谨的信息抽取器。请只根据下面的答案提取 atomic factual claims。

要求：
- 只使用 answer 本身，不要使用问题、上下文或外部知识。
- 不判断 claim 是否真实。
- 不判断 claim 是否被上下文支持。
- 只提取 answer 中明确出现的事实断言。
- 不提取寒暄、格式说明、主观评价、重复内容。
- 每个 claim 必须是独立、最小、可验证的事实断言。
- 只返回 JSON，不要返回 Markdown 或额外解释。

返回格式：
{{
  "claims": [
    {{
      "claim_id": "c1",
      "claim": "..."
    }}
  ]
}}

如果没有事实断言，返回：
{{
  "claims": []
}}

answer:
{answer}"""


def build_claim_support_prompt(context: str, claims: list[dict]) -> str:
    """构造判断事实断言是否被上下文支持的 prompt。

    Args:
        context: RAG 生成时使用的检索上下文。
        claims: 从生成答案中抽取出的事实断言。

    Returns:
        要求 LLM 仅以 JSON 返回支持性判断的 prompt 文本。
    """
    claims_json = json.dumps(claims, ensure_ascii=False, indent=2)
    return f"""你是一个严格的 RAG 忠实度评估器。请根据 retrieved context 判断每个 claim 是否被支持。

要求：
- 只能使用 context，不允许使用外部知识。
- 如果 claim 在真实世界中正确，但 context 没有支持，也必须标为 UNSUPPORTED。
- 不允许修改 claim 文本。
- 不允许新增 claim。
- 只能判断已有 claims。
- 只返回 JSON，不要返回 Markdown 或额外解释。

标签：
- SUPPORTED：context 明确支持该断言。
- PARTIALLY_SUPPORTED：context 部分支持，但断言有扩展、不完整或需要轻微推断。
- UNSUPPORTED：context 不支持该断言，或与 context 矛盾。

返回格式：
{{
  "judgements": [
    {{
      "claim_id": "c1",
      "label": "SUPPORTED",
      "evidence": "支持该 claim 的简短依据或 chunk_id；如果没有则为 null",
      "reason": "简短原因"
    }}
  ]
}}

context:
{context}

claims:
{claims_json}"""


def build_answer_to_questions_prompt(answer: str) -> str:
    """构造根据答案反推可能问题的 prompt。

    Args:
        answer: 生成答案文本。

    Returns:
        要求 LLM 仅以 JSON 返回反推问题的 prompt 文本。
    """
    return f"""你是一个严谨的问题反推器。请只根据下面的 answer，反推 1-3 个该答案可能在回答的问题。

要求：
- 只根据 answer 本身生成问题。
- 不提供 original question。
- 不判断事实正确性。
- 不判断是否基于 context。
- 问题应覆盖 answer 的主要语义。
- 只返回 JSON，不要返回 Markdown 或额外解释。

返回格式：
{{
  "generated_questions": [
    "..."
  ]
}}

answer:
{answer}"""


def build_answer_relevance_judgement_prompt(
    question: str,
    generated_questions: list[str],
) -> str:
    """构造通过反推问题判断答案相关性的 prompt。

    Args:
        question: 原始用户问题。
        generated_questions: 仅根据生成答案反推出的问题列表。

    Returns:
        要求 LLM 仅以 JSON 返回语义相关性判断的 prompt 文本。
    """
    generated_questions_json = json.dumps(
        generated_questions,
        ensure_ascii=False,
        indent=2,
    )
    return f"""你是一个严格的答案相关性评估器。请判断 original question 是否与 generated_questions 语义匹配。

要求：
- 不判断 answer 的事实正确性。
- 不判断 answer 是否来自 context。
- 只判断 answer 是否对准了 original question。
- label 只能是 RELEVANT、PARTIALLY_RELEVANT、IRRELEVANT。
- score 只能使用 1.0、0.5、0.0。
- 只返回 JSON，不要返回 Markdown 或额外解释。

评分：
- 1.0：generated_questions 与 original question 语义高度一致，说明 answer 直接回答了问题。
- 0.5：generated_questions 与 original question 部分相关，说明 answer 只部分回答或有所偏移。
- 0.0：generated_questions 与 original question 基本不相关，说明 answer 没有回答问题。

返回格式：
{{
  "score": 1.0,
  "label": "RELEVANT",
  "reason": "简短原因"
}}

original question:
{question}

generated_questions:
{generated_questions_json}"""
