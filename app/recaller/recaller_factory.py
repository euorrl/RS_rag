from typing import Any

from app.embedder import BaseEmbedder
from app.schemas import RetrievedChunk
from app.vector_store import BaseVectorStore

from app.recaller.recaller_base import BaseRecaller
from app.recaller.vector_recaller import VectorRecaller


def get_recaller(
    provider: str = "vector",
    *,
    embedder: BaseEmbedder | None = None,
    vector_store: BaseVectorStore | None = None,
    embedder_provider: str = "bge",
    vector_store_provider: str = "milvus",
    embedder_kwargs: dict[str, Any] | None = None,
    vector_store_kwargs: dict[str, Any] | None = None,
) -> BaseRecaller:
    """根据 provider 获取对应的 Recaller 实例。

    Args:
        provider (str): 召回器类型。
            当前支持：
            - "vector"
        embedder (BaseEmbedder | None): 已初始化的向量化器。
            如果不传，则根据 embedder_provider 和 embedder_kwargs 自动创建。
        vector_store (BaseVectorStore | None): 已初始化的向量库。
            如果不传，则根据 vector_store_provider 和 vector_store_kwargs 自动创建。
        embedder_provider (str): 自动创建 Embedder 时使用的 provider，默认 "bge"。
            当前支持：
            - "bge"
            - "minilm"
        vector_store_provider (str): 自动创建 VectorStore 时使用的 provider，默认 "milvus"。
            当前支持：
            - "milvus"
        embedder_kwargs (dict[str, Any] | None): 传递给 get_embedder() 的参数，例如：
            - model_name (str): 使用的 embedding 模型名称
            - normalize_embeddings (bool): 是否归一化向量
            - use_instruction (bool): 是否使用 instruction prefix
        vector_store_kwargs (dict[str, Any] | None): 传递给 get_vector_store() 的参数，例如：
            - collection_name (str): collection 名称
            - host (str): 向量数据库服务地址
            - port (str): 向量数据库服务端口
            - dimension (int): 向量维度

    Returns:
        BaseRecaller: 对应的 Recaller 实例。

    Raises:
        TypeError: provider 不是字符串时抛出。
        ValueError: provider 不支持时抛出。
    """
    if not isinstance(provider, str):
        raise TypeError("provider 必须是 str 类型")

    provider = provider.lower()

    if provider == "vector":
        if embedder is None:
            from app.embedder import get_embedder

            embedder = get_embedder(
                provider=embedder_provider,
                **(embedder_kwargs or {}),
            )

        if vector_store is None:
            from app.vector_store import get_vector_store

            vector_store = get_vector_store(
                provider=vector_store_provider,
                **(vector_store_kwargs or {}),
            )

        return VectorRecaller(
            embedder=embedder,
            vector_store=vector_store,
        )

    raise ValueError(f"Unsupported recaller provider: {provider}")


def recall(
    query: str,
    provider: str = "vector",
    top_k: int = 30,
    score_threshold: float | None = 0.4,
    **kwargs,
) -> list[RetrievedChunk]:
    """根据 query 执行召回并返回候选 chunks。

    该函数是 Recaller 层的轻量封装，用于简化调用流程。
    1. 根据 provider 选择具体 Recaller。
    2. 将 query 转换为召回器可处理的查询输入。
    3. 调用 Recaller.recall() 获取候选 chunks。
    4. 根据 score_threshold 过滤低分候选结果。
    5. 返回统一的 RetrievedChunk 结构列表。

    适用于：
    - 快速调用（无需显式实例化）
    - 需要简单配置但不想直接操作类的场景
    - 上层 pipeline 中需要一次性完成召回的场景

    Args:
        query (str): 用户查询文本。
        provider (str): 召回器类型，默认 "vector"。
        top_k (int): 返回的最大候选 chunk 数量，默认 30。
        score_threshold (float | None): 最低召回分数阈值。
            默认 0.4，会过滤掉 score 低于该阈值的候选结果；如果为 None，
            则不过滤分数。
        **kwargs: 传递给 get_recaller() 的参数，例如：
            - embedder (BaseEmbedder): 已初始化的向量化器
            - vector_store (BaseVectorStore): 已初始化的向量库
            - embedder_provider (str): 自动创建 Embedder 时使用的 provider
            - vector_store_provider (str): 自动创建 VectorStore 时使用的 provider
            - embedder_kwargs (dict): Embedder 初始化参数
            - vector_store_kwargs (dict): VectorStore 初始化参数

    Returns:
        list[RetrievedChunk]: 召回命中的候选 chunks。

    Raises:
        TypeError:
            - query 不是 str
            - provider 不是 str
            - score_threshold 不是数字
            - 依赖对象类型不正确
        ValueError:
            - query 为空字符串
            - top_k 不大于 0
            - provider 不支持
    """
    recaller = get_recaller(provider=provider, **kwargs)
    return recaller.recall(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
    )
