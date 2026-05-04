from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class Document:
    """统一的文档数据结构（Document）。

    Document 表示经过 Reader 解析后的原始文档，是 RAG pipeline 的输入数据载体。

    一个 Document 通常对应一个完整文件（如 PDF、Markdown、图片解析结果等），
    在后续流程中会被切分为多个 Chunk，用于向量化和检索。

    包含：
    - 原始文本内容（text）
    - 文件来源信息（路径、文件名、类型）
    - 元信息（metadata）
    - 唯一标识（document_id）

    metadata 通常包含：
    - reader 类型（如 TextReader / MarkdownReader / MinerUReader）
    - 原始数据格式（source_format / text_format）
    - 字符数统计（char_count）
    等信息。
    """

    source_path: str
    """文档来源路径（原始文件路径）。"""

    file_name: str
    """文件名（不含路径）。"""

    file_type: str
    """文件类型（如 .pdf / .md / .txt）。"""

    text: str
    """文档解析后的文本内容（通常为 Markdown 格式）。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """文档元信息。

    用于存储 Reader 阶段附加的信息，例如：
    - reader 类型
    - 原始格式（source_format / text_format）
    - 字符数（char_count）
    等。
    """

    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """文档唯一 ID（自动生成）。

    用于在 Chunk、Embedding、检索结果中追踪文档来源。
    """
