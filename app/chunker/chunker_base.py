from abc import ABC, abstractmethod

from app.schemas import Document, Chunk


class BaseChunker(ABC):
    """Chunker 抽象基类。

    所有具体 Chunker 都必须实现 chunk 方法。
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """将 Document 切分为标准化的 Chunk 列表。

        Args:
            document (Document): Reader 层输出的标准化文档对象。

        Returns:
            list[Chunk]: 切分后的 Chunk 对象列表。
        """
        pass  # pragma: no cover
