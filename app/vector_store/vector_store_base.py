from abc import ABC, abstractmethod

from app.schemas import EmbeddedChunk, RetrievedChunk


class BaseVectorStore(ABC):
    """向量数据库抽象基类。"""

    @abstractmethod
    def create_collection(self) -> None:
        """创建 collection。"""
        raise NotImplementedError

    @abstractmethod
    def insert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """插入已向量化的 chunks。"""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """根据 query 向量检索最相似的 chunks。"""
        raise NotImplementedError

    @abstractmethod
    def drop_collection(self) -> None:
        """删除 collection，主要用于测试。"""
        raise NotImplementedError
