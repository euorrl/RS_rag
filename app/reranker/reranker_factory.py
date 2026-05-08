from app.reranker.bge_reranker import BGEReranker
from app.reranker.reranker_base import BaseReranker
from app.schemas import RetrievedChunk


def get_reranker(
    provider: str = "bge",
    **kwargs,
) -> BaseReranker:
    """根据 provider 获取对应的 Reranker 实例。

    Args:
        provider (str): 重排序器类型。
            当前支持：
            - "bge"
        **kwargs: 传递给具体 Reranker 的初始化参数，例如：
            - model_name (str): 使用的 reranker 模型名称

    Returns:
        BaseReranker: 对应的 Reranker 实例。

    Raises:
        TypeError: provider 不是字符串时抛出。
        ValueError: provider 不支持时抛出。
    """
    if not isinstance(provider, str):
        raise TypeError("provider 必须是 str 类型")

    provider = provider.lower()

    if provider == "bge":
        return BGEReranker(**kwargs)

    raise ValueError(f"Unsupported reranker provider: {provider}")


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    provider: str = "bge",
    top_n: int = 10,
    score_threshold: float | None = None,
    **kwargs,
) -> list[RetrievedChunk]:
    """对候选 RetrievedChunk 列表执行重排序。

    该函数是 Reranker 层的轻量封装，用于简化调用流程。
    1. 根据 provider 选择具体 Reranker。
    2. 将 query 和 candidates 传递给 Reranker.rerank()。
    3. 根据 score_threshold 过滤低分候选结果。
    4. 返回重排后的 RetrievedChunk 列表。

    适用于：
    - 快速调用（无需显式实例化）
    - 需要简单配置但不想直接操作类的场景
    - 上层 pipeline 中需要一次性完成重排序的场景

    Args:
        query (str): 用户查询文本。
        candidates (list[RetrievedChunk]): 待重排的候选 chunks。
        provider (str): 重排序器类型，默认 "bge"。
        top_n (int): 返回的最大结果数量，默认 10。
        score_threshold (float | None): 最低重排分数阈值。
            如果为 None，则不过滤分数；如果为数字，则过滤掉 score
            低于该阈值的候选结果。
        **kwargs: 传递给具体 Reranker 的初始化参数，例如：
            - model_name (str): 使用的 reranker 模型名称

    Returns:
        list[RetrievedChunk]: 重排后的候选 chunks。

    Raises:
        TypeError:
            - query 不是 str
            - candidates 不是 list[RetrievedChunk]
            - provider 不是 str
            - score_threshold 不是数字
        ValueError:
            - query 为空字符串
            - top_n 不大于 0
            - provider 不支持
    """
    reranker = get_reranker(provider=provider, **kwargs)
    return reranker.rerank(
        query=query,
        candidates=candidates,
        top_n=top_n,
        score_threshold=score_threshold,
    )
