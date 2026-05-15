"""RAG 生成阶段评估 pipeline。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluation.generation import (
    evaluate_answer_relevance,
    evaluate_claim_faithfulness,
)

DEFAULT_DATASET_PATH = Path("evaluation/dataset/eval_dataset.json")

GenerateAnswerFn = Callable[[str], dict[str, Any]]


def build_context_from_chunks(retrieved_chunks: list[Any]) -> str:
    """从 retrieved chunks 拼接生成评估所需 context。

    Args:
        retrieved_chunks: 用于构造 context 的 chunk 对象或 dict 列表。

    Returns:
        使用 chunk_id 和 text 拼接得到的 context 文本。
    """
    context_blocks = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
        else:
            chunk_id = getattr(chunk, "chunk_id", "")
            text = getattr(chunk, "text", "")

        context_blocks.append(f"[chunk_id: {chunk_id}]\n{text}")

    return "\n\n".join(context_blocks)


def load_eval_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    """读取生成阶段评估数据集。

    Args:
        dataset_path: eval dataset 的 JSON 文件路径。

    Returns:
        评估样本列表。

    Raises:
        TypeError: 当 JSON 顶层结构不是 list 时抛出。
    """
    with Path(dataset_path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError("eval dataset 必须是 list[dict] 格式")

    return data


def build_strict_rag_prompt(question: str, context: str) -> str:
    """构造严格基于 context 作答的 RAG prompt。

    Args:
        question: 用户问题。
        context: 本次生成答案使用的检索上下文。

    Returns:
        用于生成严格 RAG 答案的字符串 prompt。
    """
    return f"""你是一个严格基于资料库回答的 RAG 助手。
请只根据给定 context 回答问题。
不要使用外部知识。
如果 context 中没有足够信息，请明确说明给定资料不足。
不要编造 context 中没有支持的事实。

Question:
{question}

Context:
{context}

Answer:"""


def build_generation_pipeline() -> GenerateAnswerFn:
    """构建默认 strict RAG 生成 pipeline。

    使用 app 中已有 recaller、reranker 和 generator 串联生成答案。

    Returns:
        输入问题并返回 answer、context 和 retrieved_chunks 的生成函数。
    """
    from app.generator import generate
    from app.recaller import get_recaller
    from app.reranker import get_reranker

    recaller = get_recaller()
    reranker = get_reranker()

    def generate_answer(question: str) -> dict[str, Any]:
        recalled_chunks = recaller.recall(query=question)
        reranked_chunks = reranker.rerank(
            query=question,
            candidates=recalled_chunks,
        )
        context = build_context_from_chunks(reranked_chunks)
        prompt = build_strict_rag_prompt(question=question, context=context)
        answer = generate(prompt, model="gpt-5.5")

        return {
            "answer": answer,
            "context": context,
            "retrieved_chunks": reranked_chunks,
        }

    return generate_answer


def _extract_retrieved_ids(retrieved_chunks: list[Any]) -> list[str]:
    """从 retrieved chunks 中提取 chunk_id。

    Args:
        retrieved_chunks: 用于构造 context 的 chunk 对象或 dict 列表。

    Returns:
        可提取出的 chunk_id 列表。
    """
    retrieved_ids = []
    for chunk in retrieved_chunks:
        chunk_id = (
            chunk.get("chunk_id")
            if isinstance(chunk, dict)
            else getattr(chunk, "chunk_id", None)
        )
        if chunk_id:
            retrieved_ids.append(str(chunk_id))

    return retrieved_ids


def evaluate_generation(
    eval_dataset: list[dict[str, Any]],
    generate_answer: GenerateAnswerFn,
) -> dict[str, Any]:
    """逐样本运行生成阶段评估并汇总平均指标。

    Args:
        eval_dataset: 评估样本列表，每条样本至少包含 id 和 question。
        generate_answer: 外部传入的生成函数，返回 answer/context/chunks。

    Returns:
        包含逐样本结果和整体平均指标的字典。
    """
    items = []
    claim_faithfulness_scores: list[float] = []
    answer_relevance_scores: list[float] = []

    for sample in eval_dataset:
        question = sample["question"]
        generation_result = generate_answer(question)
        answer = generation_result.get("answer", "")
        context = generation_result.get("context", "")
        retrieved_chunks = generation_result.get("retrieved_chunks", [])
        if not context and retrieved_chunks:
            context = build_context_from_chunks(retrieved_chunks)

        faithfulness_result = evaluate_claim_faithfulness(
            context=context,
            answer=answer,
        )
        relevance_result = evaluate_answer_relevance(
            question=question,
            answer=answer,
        )

        claim_faithfulness_scores.append(
            faithfulness_result.get("claim_faithfulness", 0.0)
        )
        answer_relevance_scores.append(relevance_result.get("score", 0.0))

        items.append(
            {
                "id": sample.get("id"),
                "question": question,
                "golden_chunks": sample.get("chunks", []),
                "retrieved_ids": _extract_retrieved_ids(retrieved_chunks),
                "answer": answer,
                "claim_faithfulness": faithfulness_result,
                "answer_relevance": relevance_result,
            }
        )

    total = len(eval_dataset)
    mean_claim_faithfulness = sum(claim_faithfulness_scores) / total if total else 0.0
    mean_answer_relevance = sum(answer_relevance_scores) / total if total else 0.0

    return {
        "total": total,
        "mean_claim_faithfulness": mean_claim_faithfulness,
        "mean_answer_relevance": mean_answer_relevance,
        "items": items,
    }


def run_generation_pipeline(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    generate_answer: GenerateAnswerFn | None = None,
) -> dict[str, Any]:
    """运行完整生成阶段评估 pipeline。

    Args:
        dataset_path: 待评估数据集路径。
        generate_answer: 外部传入的生成函数。

    Returns:
        生成阶段评估结果。

        如果不传，则自动构建默认 strict RAG 生成 pipeline。
    """
    eval_dataset = load_eval_dataset(dataset_path)
    generation_pipeline = generate_answer or build_generation_pipeline()
    return evaluate_generation(eval_dataset, generation_pipeline)
