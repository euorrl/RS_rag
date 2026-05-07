from numbers import Real

from app.embedder import BaseEmbedder
from app.schemas import RetrievedChunk
from app.vector_store import BaseVectorStore

from app.recaller.recaller_base import BaseRecaller


class VectorRecaller(BaseRecaller):
    """纯向量召回器。

    VectorRecaller 负责将 query 转换为向量，并调用向量库执行相似度搜索。
    该类只处理一阶段向量召回，不包含 BM25、hybrid 或 reranker 逻辑。

    Attributes:
        embedder (BaseEmbedder): 用于生成 query embedding 的向量化器。
        vector_store (BaseVectorStore): 用于执行向量搜索的向量数据库封装。
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ) -> None:
        """初始化 VectorRecaller。

        Args:
            embedder (BaseEmbedder): 已初始化的向量化器。
            vector_store (BaseVectorStore): 已初始化的向量库实例。

        Raises:
            TypeError: embedder 或 vector_store 类型不正确时抛出。
        """
        if not isinstance(embedder, BaseEmbedder):
            raise TypeError("embedder 必须是 BaseEmbedder 类型")

        if not isinstance(vector_store, BaseVectorStore):
            raise TypeError("vector_store 必须是 BaseVectorStore 类型")

        self.embedder = embedder
        self.vector_store = vector_store

    def recall(
        self,
        query: str,
        top_k: int = 30,
        score_threshold: float | None = 0.4,
    ) -> list[RetrievedChunk]:
        """执行纯向量召回。

        处理流程：
        1. 校验 query、top_k 和 score_threshold。
        2. 使用 embedder.embed_query() 生成 query 向量。
        3. 调用 vector_store.search() 召回相似 chunks。
        4. 根据 score_threshold 过滤低分结果。
        5. 补充 rank、retrieval_method 和 score_details 等统一结果字段。

        Args:
            query (str): 用户查询文本。
            top_k (int): 返回的最大候选 chunk 数量，默认返回 30 条。
            score_threshold (float | None): 最低召回分数阈值。
                默认 0.4，只保留 score 大于等于该阈值的结果；如果为 None，
                则不过滤分数。

        Returns:
            list[RetrievedChunk]: 纯向量召回结果。若没有结果满足阈值，返回空列表。

        Raises:
            TypeError:
                - query 不是 str
                - score_threshold 不是数字
                - vector_store.search() 返回值不是 list[RetrievedChunk]
            ValueError:
                - query 为空字符串
                - top_k 不大于 0
        """
        # =========================
        # 输入校验
        # =========================
        if not isinstance(query, str):
            raise TypeError("query 必须是 str 类型")

        if not query.strip():
            raise ValueError("query 不能为空字符串")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if score_threshold is not None and not isinstance(score_threshold, Real):
            raise TypeError("score_threshold 必须是数字或 None")

        # =========================
        # query 向量化并执行向量搜索
        # =========================
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vector, top_k=top_k)

        # =========================
        # 输出结果校验
        # =========================
        if not isinstance(results, list):
            raise TypeError("vector_store.search 必须返回 list[RetrievedChunk]")

        if not all(isinstance(result, RetrievedChunk) for result in results):
            raise TypeError("召回结果必须全部是 RetrievedChunk 类型")

        results = self._filter_by_score(results, score_threshold)
        return self._normalize_results(results)

    def _filter_by_score(
        self,
        results: list[RetrievedChunk],
        score_threshold: float | None,
    ) -> list[RetrievedChunk]:
        """根据分数阈值过滤召回结果。

        Args:
            results (list[RetrievedChunk]): 向量库返回的原始召回结果。
            score_threshold (float | None): 最低召回分数阈值。

        Returns:
            list[RetrievedChunk]: 过滤后的召回结果。
        """
        if score_threshold is None:
            return results

        return [result for result in results if result.score >= score_threshold]

    def _normalize_results(
        self,
        results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """补充统一召回结果字段。

        Args:
            results (list[RetrievedChunk]): 过滤后的召回结果。

        Returns:
            list[RetrievedChunk]: 补充统一字段后的召回结果。
        """
        for index, result in enumerate(results, start=1):
            result.rank = index
            result.retrieval_method = "vector"

            if "vector_score" not in result.score_details:
                result.score_details["vector_score"] = result.score

        return results
