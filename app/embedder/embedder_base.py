from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Embedder 抽象基类。

    所有具体 Embedder 都必须实现 embed_texts 方法。
    如果该 Embedder 支持检索查询，也可以实现 embed_query 方法。
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表。

        Args:
            texts (list[str]): 待向量化的文本列表。

        Returns:
            list[list[float]]: 每个文本对应一个向量。
        """
        pass  # pragma: no cover

    def embed_query(self, query: str) -> list[float]:
        """将查询文本转换为向量。

        默认实现适用于对称 embedding 模型。
        非对称模型（如 BGE）应重写该方法。

        Args:
            query (str): 查询文本。

        Returns:
            list[float]: 查询向量。
        """
        return self.embed_texts([query])[0]
