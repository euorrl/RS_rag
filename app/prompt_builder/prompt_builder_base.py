from abc import ABC, abstractmethod

from app.schemas import RetrievedChunk

Prompt = str | list[dict[str, str]]


class BasePromptBuilder(ABC):
    """PromptBuilder 抽象基类。

    PromptBuilder 是 RAG pipeline 中位于 reranker 之后、generator 之前的组件，
    只负责将用户 query 和候选 RetrievedChunk 组织成可交给 LLM 的 prompt。

    PromptBuilder 不负责：
    - 召回候选 chunk
    - 对候选 chunk 重新排序
    - 调用 LLM 生成答案

    后续如果需要不同的 prompt 模板、引用格式或多轮对话格式，可以继承该基类实现。
    """

    @abstractmethod
    def build(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> Prompt:
        """根据 query 和候选 RetrievedChunk 构建 prompt。

        Args:
            query (str): 用户查询文本。
            retrieved_chunks (list[RetrievedChunk]): 已召回或已重排的候选 chunks。
            history (list[dict[str, str]] | None): 可选历史对话消息。

        Returns:
            Prompt: 构建完成的 prompt，可以是字符串或 chat messages。
        """
        pass  # pragma: no cover
