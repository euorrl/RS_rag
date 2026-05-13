"""RAG 召回阶段评估 pipeline。"""

from __future__ import annotations

import json
from time import perf_counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluation.retrieval import evidence_recall_at_k, mrr_at_k

DEFAULT_DATASET_PATH = Path("evaluation/dataset/eval_dataset.json")
DEFAULT_K = 10
DEFAULT_RECALL_TOP_K = 30
DEFAULT_RECALL_SCORE_THRESHOLD = 0.4
DEFAULT_RERANK_TOP_N = 10
DEFAULT_RERANK_SCORE_THRESHOLD = 0.5

RetrieveFn = Callable[[str], list[Any]]


def load_eval_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    """读取召回评估数据集。

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


def build_retrieval_pipeline() -> RetrieveFn:
    """构建真实 app 召回与重排 pipeline。

    使用 app 中的默认 recaller 和 reranker：
    1. 对问题执行向量召回。
    2. 对召回结果执行重排。
    3. 返回重排后的 chunk 列表。

    Returns:
        可调用的检索 pipeline，输入问题文本，返回重排后的 RetrievedChunk 列表。
    """
    from app.recaller import get_recaller
    from app.reranker import get_reranker

    recaller_started_at = perf_counter()
    recaller = get_recaller()
    recaller_init_seconds = perf_counter() - recaller_started_at

    reranker_started_at = perf_counter()
    reranker = get_reranker()
    reranker_init_seconds = perf_counter() - reranker_started_at

    def retrieve(question: str) -> list[Any]:
        recall_started_at = perf_counter()
        recalled_chunks = recaller.recall(
            query=question,
            top_k=DEFAULT_RECALL_TOP_K,
            score_threshold=DEFAULT_RECALL_SCORE_THRESHOLD,
        )
        recall_seconds = perf_counter() - recall_started_at

        rerank_started_at = perf_counter()
        reranked_chunks = reranker.rerank(
            query=question,
            candidates=recalled_chunks,
            top_n=DEFAULT_RERANK_TOP_N,
            score_threshold=DEFAULT_RERANK_SCORE_THRESHOLD,
        )
        rerank_seconds = perf_counter() - rerank_started_at
        retrieve.last_timing = {
            "recall_seconds": recall_seconds,
            "rerank_seconds": rerank_seconds,
        }
        return reranked_chunks

    retrieve.initialization_timing = {
        "recaller_init_seconds": recaller_init_seconds,
        "reranker_init_seconds": reranker_init_seconds,
        "pipeline_init_seconds": recaller_init_seconds + reranker_init_seconds,
    }
    retrieve.last_timing = None
    return retrieve


def evaluate_retrieval(
    eval_dataset: list[dict[str, Any]],
    retrieve: RetrieveFn,
) -> dict[str, Any]:
    """逐样本计算召回率和 MRR，并汇总平均值。

    Args:
        eval_dataset: 评估样本列表，每条样本包含 id、question 和 chunks。
        retrieve: 召回与重排 pipeline。

    Returns:
        包含逐样本指标和整体平均指标的字典。
    """
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    retrieval_seconds: list[float] = []
    recall_seconds: list[float] = []
    rerank_seconds: list[float] = []

    for sample in eval_dataset:
        question = sample["question"]
        golden_chunks = sample.get("chunks", [])
        retrieval_started_at = perf_counter()
        retrieved_chunks = retrieve(question)
        retrieval_seconds.append(perf_counter() - retrieval_started_at)
        retrieved_ids = [chunk.chunk_id for chunk in retrieved_chunks]

        stage_timing = getattr(retrieve, "last_timing", None)
        if isinstance(stage_timing, dict):
            recall_time = stage_timing.get("recall_seconds")
            rerank_time = stage_timing.get("rerank_seconds")
            if isinstance(recall_time, (int, float)):
                recall_seconds.append(float(recall_time))
            if isinstance(rerank_time, (int, float)):
                rerank_seconds.append(float(rerank_time))

        recall = evidence_recall_at_k(
            retrieved_ids=retrieved_ids,
            golden_chunks=golden_chunks,
            k=DEFAULT_K,
        )
        mrr = mrr_at_k(
            retrieved_ids=retrieved_ids,
            golden_chunks=golden_chunks,
            k=DEFAULT_K,
        )

        recall_scores.append(recall)
        mrr_scores.append(mrr)

    total = len(eval_dataset)
    mean_recall = sum(recall_scores) / total if total else 0.0
    mean_mrr = sum(mrr_scores) / total if total else 0.0
    mean_retrieval_seconds = sum(retrieval_seconds) / total if total else 0.0

    result = {
        "total": total,
        "k": DEFAULT_K,
        "recall_top_k": DEFAULT_RECALL_TOP_K,
        "recall_score_threshold": DEFAULT_RECALL_SCORE_THRESHOLD,
        "rerank_top_n": DEFAULT_RERANK_TOP_N,
        "rerank_score_threshold": DEFAULT_RERANK_SCORE_THRESHOLD,
        f"evidence_recall@{DEFAULT_K}": recall_scores,
        f"mean_evidence_recall@{DEFAULT_K}": mean_recall,
        f"mrr@{DEFAULT_K}": mrr_scores,
        f"mean_mrr@{DEFAULT_K}": mean_mrr,
        "retrieval_seconds": retrieval_seconds,
        "mean_retrieval_seconds": mean_retrieval_seconds,
    }

    initialization_timing = getattr(retrieve, "initialization_timing", None)
    if isinstance(initialization_timing, dict):
        result["initialization_timing_seconds"] = initialization_timing

    if recall_seconds:
        result["recall_seconds"] = recall_seconds
        result["mean_recall_seconds"] = sum(recall_seconds) / len(recall_seconds)

    if rerank_seconds:
        result["rerank_seconds"] = rerank_seconds
        result["mean_rerank_seconds"] = sum(rerank_seconds) / len(rerank_seconds)

    return result


def run_retrieval_pipeline(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    retrieve: RetrieveFn | None = None,
) -> dict[str, Any]:
    """运行完整召回评估 pipeline。

    Args:
        dataset_path: 待评估数据集路径。
        retrieve: 可选的外部检索函数，主要用于单元测试。

    Returns:
        召回评估结果。
    """
    eval_dataset = load_eval_dataset(dataset_path)
    retrieval_pipeline = retrieve or build_retrieval_pipeline()
    return evaluate_retrieval(eval_dataset, retrieval_pipeline)
