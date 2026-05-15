"""RAG 生成阶段评估的纯指标函数。"""

from __future__ import annotations


def compute_claim_faithfulness(labels: list[str]) -> float:
    """根据事实断言支持标签计算答案忠实度。

    Args:
        labels: 事实断言支持标签。支持 ``SUPPORTED``、
            ``PARTIALLY_SUPPORTED`` 和 ``UNSUPPORTED``。

    Returns:
        平均忠实度分数。输入为空时返回 ``0.0``。
    """
    if not labels:
        return 0.0

    label_scores = {
        "SUPPORTED": 1.0,
        "PARTIALLY_SUPPORTED": 0.5,
        "UNSUPPORTED": 0.0,
    }
    scores = [label_scores.get(label.strip().upper(), 0.0) for label in labels]
    return sum(scores) / len(scores)


def normalize_answer_relevance_score(
    score: float | int | None,
    label: str | None,
) -> float:
    """规范化答案相关性分数，优先使用合法 score，其次使用 label。

    Args:
        score: LLM 返回的原始分数。
        label: LLM 返回的原始相关性标签。

    Returns:
        ``{0.0, 0.5, 1.0}`` 中的规范化分数。无法判断时返回 ``0.0``。
    """
    if score in {0.0, 0.5, 1.0}:
        return float(score)

    label_scores = {
        "RELEVANT": 1.0,
        "PARTIALLY_RELEVANT": 0.5,
        "IRRELEVANT": 0.0,
    }
    if label is None:
        return 0.0

    return label_scores.get(label.strip().upper(), 0.0)
