from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddedChunk:
    """带向量信息的文本块（EmbeddedChunk）。

    EmbeddedChunk 是 Chunk 经过 Embedder 处理后的结果，
    是写入向量数据库和执行向量检索的核心数据结构。

    可以理解为：
    Chunk + embedding

    包含：
    - chunk_id：对应原始 Chunk
    - document_id：来源文档
    - text：原始文本
    - embedding：向量表示
    - metadata：上下文信息
    """

    chunk_id: str
    """对应 Chunk 的唯一 ID。"""

    document_id: str | None
    """所属 Document 的 ID，用于追溯来源。"""

    text: str
    """原始文本内容（用于检索返回）。"""

    embedding: list[float]
    """向量表示（embedding）。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """元信息。

    通常继承自 Chunk.metadata，并可扩展，例如：
    - source_path
    - file_name
    - h1 / h2 / h3
    - chunk_index
    - chunk_size
    - embedding_model
    """
