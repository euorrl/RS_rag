from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class Document:
    """统一的文档数据结构。"""

    source_path: str
    file_name: str
    file_type: str
    text: str

    metadata: dict[str, Any] = field(default_factory=dict)

    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
