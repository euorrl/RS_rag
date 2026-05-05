from app.embedder.bge_embedder import BGEEmbedder
from app.embedder.embedder_base import BaseEmbedder
from app.embedder.minilm_embedder import SentenceTransformerEmbedder
from app.schemas import Chunk, EmbeddedChunk


def get_embedder(
    provider: str = "bge",
    **kwargs,
) -> BaseEmbedder:
    """根据 provider 获取对应的 Embedder 实例。

    Args:
        provider (str): 向量化模型类型。
            当前支持：
            - "bge"
            - "minilm"
        **kwargs: 传递给具体 Embedder 的初始化参数，例如：
            - model_name (str): 使用的 embedding 模型名称
            - normalize_embeddings (bool): 是否归一化向量
            - use_instruction (bool): 是否使用 instruction prefix

    Returns:
        BaseEmbedder: 对应的 Embedder 实例。

    Raises:
        TypeError: provider 不是字符串时抛出。
        ValueError: provider 不支持时抛出。
    """
    if not isinstance(provider, str):
        raise TypeError("provider 必须是 str 类型")

    provider = provider.lower()

    if provider == "bge":
        return BGEEmbedder(**kwargs)

    if provider == "minilm":
        return SentenceTransformerEmbedder(**kwargs)

    raise ValueError(f"Unsupported embedder provider: {provider}")


def embed_chunks(
    chunks: list[Chunk],
    provider: str = "bge",
    **kwargs,
) -> list[EmbeddedChunk]:
    """将 Chunk 列表转换为 EmbeddedChunk 列表（执行向量化）。

    该函数是 Embedder 层的统一入口：
    1. 根据 provider 选择具体 Embedder。
    2. 提取 Chunk 中的文本。
    3. 批量生成 embedding。
    4. 构建 EmbeddedChunk 结构返回。

    Args:
        chunks (list[Chunk]): 待向量化的文本块列表。
        provider (str): 向量化模型类型，默认 "bge"。
        **kwargs: 传递给具体 Embedder 的初始化参数，例如：
            - model_name (str): 使用的 embedding 模型名称
            - normalize_embeddings (bool): 是否归一化向量
            - use_instruction (bool): 是否使用 instruction prefix

    Returns:
        list[EmbeddedChunk]: 包含 embedding 的文本块列表。

    Raises:
        TypeError:
            - chunks 不是 list
            - chunks 中元素不是 Chunk
            - provider 不是 str
        RuntimeError: embedding 数量与 chunk 数量不一致时抛出。
        ValueError: provider 不支持时抛出。
    """
    # =========================
    # 输入校验
    # =========================
    if not isinstance(provider, str):
        raise TypeError("provider 必须是 str 类型")

    provider = provider.lower()

    if not isinstance(chunks, list):
        raise TypeError("chunks 必须是 list[Chunk] 类型")

    if not all(isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("chunks 中的元素必须全部是 Chunk 类型")

    # 空输入直接返回
    if not chunks:
        return []

    # =========================
    # 获取 Embedder 实例
    # =========================
    embedder = get_embedder(provider=provider, **kwargs)

    # =========================
    # 提取文本并生成向量
    # =========================
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    # 防御性检查：数量必须一致
    if len(embeddings) != len(chunks):
        raise RuntimeError("embedding 数量与 chunk 数量不一致")

    # =========================
    # 构建 EmbeddedChunk
    # =========================
    embedded_chunks: list[EmbeddedChunk] = []

    for chunk, embedding in zip(chunks, embeddings):
        # 拷贝 metadata，避免污染原始 Chunk
        metadata = dict(chunk.metadata)

        # 记录 embedding provider
        metadata["embedding_provider"] = provider

        # 记录 embedding model
        model_name = getattr(embedder, "model_name", None)
        if model_name:
            metadata["embedding_model"] = model_name

        # 记录是否归一化
        normalize_embeddings = getattr(embedder, "normalize_embeddings", None)
        if normalize_embeddings is not None:
            metadata["normalize_embeddings"] = normalize_embeddings

        # 记录是否使用 instruction prefix
        use_instruction = getattr(embedder, "use_instruction", None)
        if use_instruction is not None:
            metadata["use_instruction"] = use_instruction

        embedded_chunks.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                embedding=embedding,
                metadata=metadata,
            )
        )

    return embedded_chunks
