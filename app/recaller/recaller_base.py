from abc import ABC, abstractmethod

from app.schemas import RetrievedChunk


class BaseRecaller(ABC):
    """Recaller 抽象基类。

    Recaller 是 RAG retrieval 阶段中的一阶段召回组件，负责根据用户 query
    从向量库、倒排索引或其他候选源中取回候选 chunks。

    当前项目只实现纯向量召回，后续 BM25、hybrid 等召回器可以继承该基类。
    reranker 不属于 recaller，但可以继续消费 recaller 返回的 RetrievedChunk。
    """

    @abstractmethod
    def recall(
        self,
        query: str,
        top_k: int = 30,
        score_threshold: float | None = 0.4,
    ) -> list[RetrievedChunk]:
        """根据 query 召回候选 chunks。

        Args:
            query (str): 用户查询文本。
            top_k (int): 返回的最大候选 chunk 数量，默认返回 30 条。
            score_threshold (float | None): 最低召回分数阈值。
                默认 0.4，会过滤掉 score 低于阈值的结果；如果为 None，则不过滤分数。

        Returns:
            list[RetrievedChunk]: 召回命中的候选 chunks。
        """
        pass  # pragma: no cover
