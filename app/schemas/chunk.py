from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class Chunk:
    """统一的文本块数据结构（Chunk）。

    Chunk 是 Document 经过切分后的最小处理单元，
    用于后续的向量化（embedding）、检索（retrieval）和生成（generation）。

    每个 Chunk 包含：
    - 文本内容（text）
    - 所属文档 ID（document_id）
    - 唯一标识（chunk_id）
    - 元信息（metadata）

    metadata 包含：
    - 文档来源信息（source_path、file_name 等）
    - 标题层级（h1、h2、h3）
    - chunk 位置（chunk_index）
    - chunk 长度（chunk_size）
    等上下文信息，用于提升检索质量与可解释性。
    """

    document_id: str | None
    """所属 Document 的唯一标识，用于追溯来源。"""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Chunk 的唯一 ID（自动生成）。"""

    text: str = ""
    """Chunk 的文本内容（用于 embedding 和检索）。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Chunk 的元信息。

    用于存储上下文信息，例如：
    - 文档来源（source_path、file_name）
    - 标题层级（h1/h2/h3）
    - 位置索引（chunk_index）
    - 文本长度（chunk_size）
    等。
    """
