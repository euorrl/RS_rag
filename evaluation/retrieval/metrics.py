"""RAG 召回阶段的 evidence 评估指标。"""


def evidence_recall_at_k(
    retrieved_ids: list[str],
    golden_chunks: list[list[str]],
    k: int,
) -> float:
    """计算 Evidence Recall@K。

    ``golden_chunks`` 中的每个内层列表表示一个需要覆盖的 evidence。
    如果某个内层列表中的任意 chunk_id 出现在前 k 个召回结果中，
    则认为该 evidence 被覆盖。

    Args:
        retrieved_ids: 按召回排序排列的 chunk_id 列表。
        golden_chunks: golden evidence chunk 分组。
        k: 参与计算的召回结果数量。

    Returns:
        Evidence Recall@K。当 ``golden_chunks`` 为空时返回 0.0。
    """
    if not golden_chunks:
        return 0.0

    top_k_ids = set(retrieved_ids[:k])
    covered = sum(
        1
        for evidence_group in golden_chunks
        if any(chunk_id in top_k_ids for chunk_id in evidence_group)
    )

    return covered / len(golden_chunks)


def mrr_at_k(
    retrieved_ids: list[str],
    golden_chunks: list[list[str]],
    k: int,
) -> float:
    """计算单条问题的 MRR@K。

    先将 ``golden_chunks`` 展平成相关 chunk_id 集合，然后在前 k 个召回结果中
    找到第一个命中的 golden chunk，并返回其 rank 的倒数。

    Args:
        retrieved_ids: 按召回排序排列的 chunk_id 列表。
        golden_chunks: golden evidence chunk 分组。
        k: 参与计算的召回结果数量。

    Returns:
        第一个命中结果的倒数排名。没有命中或 ``golden_chunks`` 为空时返回 0.0。
    """
    golden_ids = {
        chunk_id for evidence_group in golden_chunks for chunk_id in evidence_group
    }
    if not golden_ids:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in golden_ids:
            return 1 / rank

    return 0.0
