from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class Chunk:
    """统一的文本块数据结构。"""

    document_id: str | None
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
