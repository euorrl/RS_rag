from abc import ABC, abstractmethod

from app.schemas import RetrievedChunk


class BaseReranker(ABC):
    """Reranker 抽象基类。

    Reranker 是 RAG retrieval 阶段中的二阶段排序组件，负责对 recaller
    返回的候选 RetrievedChunk 重新打分和排序。

    Reranker 不负责召回、不负责 prompt 构建，也不负责调用 LLM。
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 10,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """根据 query 对候选 chunks 重新排序。

        Args:
            query (str): 用户查询文本。
            candidates (list[RetrievedChunk]): 待重排的候选 chunks。
            top_n (int): 返回的最大结果数量，默认返回 10 条。
            score_threshold (float | None): 最低重排分数阈值。
                如果为 None，则不过滤分数；如果为数字，则过滤掉 score
                低于该阈值的候选结果。

        Returns:
            list[RetrievedChunk]: 重排后的候选 chunks。
        """
        pass  # pragma: no cover
