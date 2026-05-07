from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """统一的检索结果数据结构（RetrievedChunk）。

    RetrievedChunk 是 recaller 和 reranker 阶段共用的结果单元，
    用于表示一次召回、混合检索或重排序后命中的文本块。

    可以理解为：
    Chunk 去掉 embedding 后，加上检索分数（score）、排序位置（rank）
    和检索来源信息（retrieval_method / score_details）。

    每个 RetrievedChunk 包含：
    - 文本块 ID（chunk_id）
    - 所属文档 ID（document_id）
    - 命中的文本内容（text）
    - 当前阶段主分数（score）
    - 元信息（metadata）
    - 排序与检索来源信息（rank、retrieval_method、score_details）

    metadata 通常继承自 Chunk.metadata，包含：
    - 文档来源信息（source_path、file_name 等）
    - 标题层级（h1、h2、h3）
    - chunk 位置（chunk_index）
    - chunk 长度（chunk_size）
    等上下文信息，用于结果展示、答案引用和可解释性。

    score_details 用于保存不同阶段或不同召回方式的细分分数，例如：
    - vector_score
    - bm25_score
    - hybrid_score
    - rerank_score
    """

    chunk_id: str
    """对应 Chunk 的唯一 ID。"""

    document_id: str | None
    """所属 Document 的唯一标识，用于追溯来源。"""

    text: str
    """命中的文本内容（用于构造上下文和返回给 LLM）。"""

    score: float
    """当前阶段用于排序的主分数。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """检索结果的元信息。

    通常继承自 Chunk.metadata，例如：
    - 文档来源（source_path、file_name）
    - 标题层级（h1/h2/h3）
    - 位置索引（chunk_index）
    - 文本长度（chunk_size）
    等。
    """

    milvus_id: int | str | None = None
    """Milvus 自动生成的内部主键。

    该字段只在结果来自 Milvus 时存在，不等同于业务侧 chunk_id。
    """

    rank: int | None = None
    """当前结果列表中的排序位置。"""

    retrieval_method: str = "vector"
    """产生该结果的检索方式。

    例如：
    - vector
    - bm25
    - hybrid
    - reranker
    """

    score_details: dict[str, Any] = field(default_factory=dict)
    """各阶段或各召回方式的细分分数。

    用于保存更多可解释的分数信息，例如：
    - vector_score
    - bm25_score
    - hybrid_score
    - rerank_score
    """
