from abc import ABC, abstractmethod
from collections.abc import Iterator

Prompt = str | list[dict[str, str]]


class BaseGenerator(ABC):
    """Generator 抽象基类。

    Generator 是 RAG pipeline 中负责调用 LLM 生成最终答案的组件，位于
    PromptBuilder 之后。它只负责：
    - 接收 prompt
    - 调用 LLM
    - 返回或流式产出答案文本

    Generator 不负责召回、重排、构建 prompt，也不负责维护聊天记忆。
    后续如果需要接入其他模型提供商，可以继承该基类实现。
    """

    @abstractmethod
    def generate(self, prompt: Prompt) -> str:
        """根据 prompt 生成完整答案。

        Args:
            prompt (Prompt): 输入给 LLM 的 prompt。
                可以是字符串 prompt，也可以是 chat messages。

        Returns:
            str: LLM 生成的完整答案。
        """
        pass  # pragma: no cover

    @abstractmethod
    def stream_generate(self, prompt: Prompt) -> Iterator[str]:
        """根据 prompt 流式生成答案。

        Args:
            prompt (Prompt): 输入给 LLM 的 prompt。
                可以是字符串 prompt，也可以是 chat messages。

        Yields:
            str: LLM 流式返回的文本片段。
        """
        pass  # pragma: no cover
