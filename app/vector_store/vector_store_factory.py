from app.vector_store.milvus_store import MilvusVectorStore
from app.vector_store.vector_store_base import BaseVectorStore
from app.schemas import EmbeddedChunk


def get_vector_store(
    provider: str = "milvus",
    **kwargs,
) -> BaseVectorStore:
    """根据 provider 获取对应的 VectorStore 实例。

    Args:
        provider (str): 向量数据库类型。
            当前支持：
            - "milvus"
        **kwargs: 传递给具体 VectorStore 的初始化参数，例如：
            - collection_name (str): collection 名称
            - host (str): 向量数据库服务地址
            - port (str): 向量数据库服务端口
            - dimension (int): 向量维度

    Returns:
        BaseVectorStore: 对应的 VectorStore 实例。

    Raises:
        TypeError: provider 不是字符串时抛出。
        ValueError: provider 不支持时抛出。
    """
    if not isinstance(provider, str):
        raise TypeError("provider 必须是 str 类型")

    provider = provider.lower()

    if provider == "milvus":
        return MilvusVectorStore(**kwargs)

    raise ValueError(f"Unsupported vector store provider: {provider}")


def save_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
    provider: str = "milvus",
    **kwargs,
) -> BaseVectorStore:
    """将 EmbeddedChunk 列表保存到向量数据库。

    该函数是 VectorStore 层的轻量封装，用于简化调用流程：
    1. 根据 provider 创建具体 VectorStore。
    2. 在未显式传入 dimension 时，根据首个 embedding 自动推断向量维度。
    3. 调用 VectorStore.insert() 写入数据。
    4. 返回 VectorStore 实例，便于后续继续执行 search 等操作。

    适用于：
    - 快速调用（无需显式实例化）
    - 需要简单配置但不想直接操作类的场景

    Args:
        embedded_chunks (list[EmbeddedChunk]): 待保存的向量化文本块列表。
        provider (str): 向量数据库类型，默认 "milvus"。
        **kwargs: 传递给具体 VectorStore 的初始化参数，例如：
            - collection_name (str): collection 名称
            - host (str): 向量数据库服务地址
            - port (str): 向量数据库服务端口
            - dimension (int): 向量维度

    Returns:
        BaseVectorStore: 已执行写入操作的 VectorStore 实例。

    Raises:
        TypeError:
            - embedded_chunks 不是 list
            - embedded_chunks 中包含非 EmbeddedChunk 元素
            - provider 不是字符串
        ValueError: provider 不支持时抛出。
    """
    if not isinstance(embedded_chunks, list):
        raise TypeError("embedded_chunks 必须是 list[EmbeddedChunk] 类型")

    if not all(isinstance(chunk, EmbeddedChunk) for chunk in embedded_chunks):
        raise TypeError("embedded_chunks 中的元素必须全部是 EmbeddedChunk 类型")

    if embedded_chunks and "dimension" not in kwargs:
        kwargs["dimension"] = len(embedded_chunks[0].embedding)

    vector_store = get_vector_store(provider=provider, **kwargs)
    vector_store.insert(embedded_chunks)

    return vector_store
