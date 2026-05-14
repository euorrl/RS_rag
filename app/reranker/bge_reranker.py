import os
from numbers import Real

from dotenv import load_dotenv

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


def _cuda_total_memory_gb() -> float | None:
    """返回当前 CUDA 设备显存大小；无法检测时返回 None。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device_index = torch.cuda.current_device()
        total_bytes = torch.cuda.get_device_properties(device_index).total_memory
        return total_bytes / (1024**3)
    except Exception:
        return None


def _has_cuda_device() -> bool:
    """判断当前 PyTorch 环境是否可用 CUDA。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_batch_size(device: str | None, batch_size: int | None) -> int:
    """解析 rerank batch_size，优先级为显式参数、环境变量、硬件默认值。"""
    if batch_size is not None:
        resolved_batch_size = batch_size
    elif os.getenv("RERANK_BATCH_SIZE"):
        resolved_batch_size = int(os.environ["RERANK_BATCH_SIZE"])
    else:
        resolved_batch_size = _default_batch_size_for_device(device)

    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    return resolved_batch_size


def _default_batch_size_for_device(device: str | None) -> int:
    """根据推理设备和显存给出保守默认 batch_size。"""
    normalized_device = (device or "").lower()
    use_cuda = normalized_device.startswith("cuda") or (
        device is None and _has_cuda_device()
    )

    if not use_cuda:
        return 32

    total_memory_gb = _cuda_total_memory_gb()
    if total_memory_gb is None:
        return 16
    if total_memory_gb <= 4:
        return 8
    if total_memory_gb <= 8:
        return 16
    if total_memory_gb <= 12:
        return 32
    return 64


class BGEReranker(BaseReranker):
    """基于 BGE reranker 的二阶段重排序器。

    BGEReranker 接收 recaller 返回的 RetrievedChunk 列表，将 query 和每个
    chunk.text 组成 pair，使用 cross-encoder reranker 重新计算相关性分数，
    再按 rerank 分数降序返回前 top_n 条结果。

    Attributes:
        model_name (str): 使用的 BGE reranker 模型名称。
        device (str | None): 模型推理设备，例如 "cuda" 或 "cpu"。
        batch_size (int): rerank 推理时每批处理的文本对数量。
        model: SentenceTransformers CrossEncoder 实例。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """初始化 BGEReranker。

        Args:
            model_name (str): 使用的 reranker 模型名称。
            device (str | None): 模型推理设备。默认从 RERANKER_DEVICE 环境变量读取；
                未设置时交给 sentence-transformers 自动选择。
            batch_size (int | None): rerank 推理批大小。默认从 RERANK_BATCH_SIZE
                环境变量读取；未设置时根据当前 CUDA 可用性和显存自动选择。CPU 默认
                使用 32；6GB 级别 CUDA 显卡默认使用 16。

        Raises:
            ValueError: batch_size 小于等于 0 时抛出。
            RuntimeError: 模型加载失败时抛出。
        """
        load_dotenv()
        self.model_name = model_name
        self.device = device or os.getenv("RERANKER_DEVICE") or None
        self.batch_size = _resolve_batch_size(self.device, batch_size)

        try:
            model_kwargs = {}
            if self.device is not None:
                model_kwargs["device"] = self.device
            self.model = _get_cross_encoder()(model_name, **model_kwargs)
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
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            device=self.device,
        )

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
