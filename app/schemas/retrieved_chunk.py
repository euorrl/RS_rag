from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """向量检索返回的文本块数据结构（RetrievedChunk）。

    RetrievedChunk 表示向量数据库检索命中的结果，是 RAG pipeline 中
    retrieval 阶段的标准输出数据结构。

    可以理解为：EmbeddedChunk 去掉 embedding，加上检索分数 score。
    """

    chunk_id: str
    """对应 Chunk 的唯一 ID。"""

    document_id: str | None
    """所属 Document 的 ID，用于追溯来源。"""

    text: str
    """检索命中的原始文本内容。"""

    score: float
    """向量数据库返回的相似度分数。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """检索结果的元信息，通常继承自入库时的 EmbeddedChunk.metadata。"""

    milvus_id: int | str | None = None
    """Milvus 自动生成的内部主键，不等同于业务侧 chunk_id。"""
