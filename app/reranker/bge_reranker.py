from numbers import Real

from app.reranker.reranker_base import BaseReranker
from app.schemas import RetrievedChunk

CrossEncoder = None


def _get_cross_encoder():
    """懒加载 CrossEncoder，避免导入 reranker 模块时加载重依赖。"""
    global CrossEncoder

    if CrossEncoder is None:
        from sentence_transformers import (
            CrossEncoder as SentenceTransformerCrossEncoder,
        )

        CrossEncoder = SentenceTransformerCrossEncoder

    return CrossEncoder


class BGEReranker(BaseReranker):
    """基于 BGE reranker 的二阶段重排序器。

    BGEReranker 接收 recaller 返回的 RetrievedChunk 列表，将 query 和每个
    chunk.text 组成 pair，使用 cross-encoder reranker 重新计算相关性分数，
    再按 rerank 分数降序返回前 top_n 条结果。

    Attributes:
        model_name (str): 使用的 BGE reranker 模型名称。
        model: SentenceTransformers CrossEncoder 实例。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        """初始化 BGEReranker。

        Args:
            model_name (str): 使用的 reranker 模型名称。

        Raises:
            RuntimeError: 模型加载失败时抛出。
        """
        self.model_name = model_name

        try:
            self.model = _get_cross_encoder()(model_name)
        except Exception as exc:
            raise RuntimeError(f"BGE reranker 模型加载失败: {model_name}") from exc

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 10,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """对候选 chunks 执行 BGE rerank。

        处理流程：
        1. 校验 query、candidates、top_n 和 score_threshold。
        2. 将 query 与每个 candidate.text 组成文本对。
        3. 使用 BGE reranker 计算相关性分数。
        4. 将分数写入 score 和 score_details["rerank_score"]。
        5. 根据 score_threshold 过滤低分结果。
        6. 按 rerank 分数降序排序，并返回前 top_n 条结果。

        Args:
            query (str): 用户查询文本。
            candidates (list[RetrievedChunk]): 待重排的候选 chunks。
            top_n (int): 返回的最大结果数量，默认返回 10 条。
            score_threshold (float | None): 最低重排分数阈值。
                如果为 None，则不过滤分数；如果为数字，则过滤掉 score
                低于该阈值的候选结果。

        Returns:
            list[RetrievedChunk]: 重排后的候选 chunks。

        Raises:
            TypeError:
                - query 不是 str
                - candidates 不是 list[RetrievedChunk]
                - score_threshold 不是数字
                - reranker 返回的分数不是数字
            ValueError:
                - query 为空字符串
                - top_n 不大于 0
            RuntimeError: reranker 返回的分数数量与候选数量不一致时抛出。
        """
        # =========================
        # 输入校验
        # =========================
        if not isinstance(query, str):
            raise TypeError("query 必须是 str 类型")

        if not query.strip():
            raise ValueError("query 不能为空字符串")

        if not isinstance(candidates, list):
            raise TypeError("candidates 必须是 list[RetrievedChunk] 类型")

        if not all(isinstance(candidate, RetrievedChunk) for candidate in candidates):
            raise TypeError("candidates 中的元素必须全部是 RetrievedChunk 类型")

        if top_n <= 0:
            raise ValueError("top_n 必须大于 0")

        if score_threshold is not None and not isinstance(score_threshold, Real):
            raise TypeError("score_threshold 必须是数字或 None")

        if not candidates:
            return []

        # =========================
        # 构造 pair 并执行 rerank
        # =========================
        pairs = [[query, candidate.text] for candidate in candidates]
        scores = self.model.predict(pairs)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        if len(scores) != len(candidates):
            raise RuntimeError("rerank 分数数量与 candidate 数量不一致")

        if not all(isinstance(score, Real) for score in scores):
            raise TypeError("rerank 分数必须全部是数字")

        # =========================
        # 写入 rerank 分数并排序
        # =========================
        reranked_results = []
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)
            candidate.rerank_method = "bge"
            candidate.score_details["rerank_score"] = float(score)
            reranked_results.append(candidate)

        reranked_results = self._filter_by_score(
            reranked_results,
            score_threshold,
        )
        reranked_results.sort(key=lambda result: result.score, reverse=True)
        reranked_results = reranked_results[:top_n]

        for index, result in enumerate(reranked_results, start=1):
            result.rank = index

        return reranked_results

    def _filter_by_score(
        self,
        results: list[RetrievedChunk],
        score_threshold: float | None,
    ) -> list[RetrievedChunk]:
        """根据分数阈值过滤重排结果。

        Args:
            results (list[RetrievedChunk]): 写入 rerank 分数后的结果。
            score_threshold (float | None): 最低重排分数阈值。

        Returns:
            list[RetrievedChunk]: 过滤后的重排结果。
        """
        if score_threshold is None:
            return results

        return [result for result in results if result.score >= score_threshold]
