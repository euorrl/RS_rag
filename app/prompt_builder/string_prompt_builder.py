from app.prompt_builder.prompt_builder_base import BasePromptBuilder
from app.schemas import RetrievedChunk


class StringPromptBuilder(BasePromptBuilder):
    """字符串 PromptBuilder。

    StringPromptBuilder 用于生成字符串 prompt，适合单轮问答、调试打印、日志记录，
    以及只支持 str 输入的模型或 completion-style 模型。

    该实现面向通用 RAG 问答场景，将用户 query 和 RetrievedChunk 列表拼接为
    结构化 prompt。Prompt 中只保留每个 chunk 的来源、标题和内容，不写入 chunk_id、
    document_id、score_details、embedding 等内部字段。
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        max_context_chars: int | None = 12000,
        allow_general_fallback: bool = True,
    ):
        """初始化字符串 PromptBuilder。

        Args:
            system_prompt (str | None): 自定义系统提示词。
                如果不传，则根据 allow_general_fallback 使用默认提示词。
            max_context_chars (int | None): 参考资料部分的最大字符数。
                默认 12000；如果为 None，则不截断参考资料。
            allow_general_fallback (bool): 参考资料不足时是否允许使用通用知识补充。
                默认 True；如果为 False，则必须严格只依据参考资料回答。

        Raises:
            TypeError:
                - system_prompt 不是 str 或 None
                - max_context_chars 不是 int 或 None
                - allow_general_fallback 不是 bool
            ValueError:
                - system_prompt 为空字符串
                - max_context_chars 不大于 0
        """
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt 必须是 str 或 None")

        if isinstance(system_prompt, str) and not system_prompt.strip():
            raise ValueError("system_prompt 不能为空字符串")

        if max_context_chars is not None and not isinstance(max_context_chars, int):
            raise TypeError("max_context_chars 必须是 int 或 None")

        if isinstance(max_context_chars, int) and max_context_chars <= 0:
            raise ValueError("max_context_chars 必须大于 0")

        if not isinstance(allow_general_fallback, bool):
            raise TypeError("allow_general_fallback 必须是 bool 类型")

        self.allow_general_fallback = allow_general_fallback
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.max_context_chars = max_context_chars

    def build(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """根据 query 和 RetrievedChunk 列表构建字符串 prompt。

        Args:
            query (str): 用户查询文本。
            retrieved_chunks (list[RetrievedChunk]): 已召回或已重排的候选 chunks。
            history (list[dict[str, str]] | None): 字符串 prompt 不使用历史消息。

        Returns:
            str: 可直接传递给 completion-style 模型的 prompt 文本。

        Raises:
            TypeError:
                - query 不是 str
                - retrieved_chunks 不是 list
                - retrieved_chunks 中元素不是 RetrievedChunk
                - history 不为 None
            ValueError: query 为空字符串时抛出。
        """
        if history is not None:
            raise TypeError("StringPromptBuilder 不支持 history 参数")

        self._validate_build_input(query, retrieved_chunks)

        return "\n\n".join(
            [
                self.system_prompt,
                self._build_user_prompt(query, retrieved_chunks),
            ]
        )

    def _get_default_system_prompt(self) -> str:
        """根据通用知识补充策略获取默认系统提示词。"""
        citation_rule = (
            "如果答案中使用了参考资料内容，必须在相关段落后标注来源，"
            "格式为：来源：资料文件名，标题：标题路径。"
        )

        if self.allow_general_fallback:
            return (
                "你是一个严谨的知识问答助手。请优先依据参考资料回答用户问题；"
                "如果参考资料足以回答，请基于资料进行回答，不要编造资料之外的内容。"
                "如果参考资料不足以完整回答，请明确说明“参考资料中没有提供足够信息”，"
                "然后可以基于你的通用知识进行补充说明，回答应准确、清晰。"
                "必须清楚标注哪些内容来自参考资料，哪些内容是通用知识补充。"
                f"{citation_rule}"
            )

        return (
            "你是一个严谨的知识问答助手。请严格依据参考资料回答用户问题，"
            "不要使用参考资料之外的通用知识进行补充或推断。"
            "如果参考资料不足以回答，请明确说明“参考资料中没有提供足够信息”。"
            f"{citation_rule}"
        )

    def _build_user_prompt(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        """构建包含参考资料、用户问题和作答要求的用户提示词。"""
        context_text = self._build_context_text(retrieved_chunks)

        return "\n\n".join(
            [
                "参考资料：",
                context_text,
                "用户问题：",
                query.strip(),
                "请基于以上要求作答。",
            ]
        )

    def _validate_build_input(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> None:
        """校验构建 prompt 时的输入参数。"""
        if not isinstance(query, str):
            raise TypeError("query 必须是 str 类型")

        if not query.strip():
            raise ValueError("query 不能为空字符串")

        if not isinstance(retrieved_chunks, list):
            raise TypeError("retrieved_chunks 必须是 list 类型")

        if not all(isinstance(chunk, RetrievedChunk) for chunk in retrieved_chunks):
            raise TypeError("retrieved_chunks 中的元素必须是 RetrievedChunk")

    def _build_context_text(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        """将 RetrievedChunk 列表转换为参考资料文本。"""
        if not retrieved_chunks:
            return "无可用参考资料。"

        context_blocks = [
            self._format_chunk(index=index, chunk=chunk)
            for index, chunk in enumerate(retrieved_chunks, start=1)
        ]
        context_text = "\n\n".join(context_blocks)

        if (
            self.max_context_chars is not None
            and len(context_text) > self.max_context_chars
        ):
            context_text = (
                context_text[: self.max_context_chars].rstrip()
                + "\n...[参考资料已截断]"
            )

        return context_text

    def _format_chunk(self, index: int, chunk: RetrievedChunk) -> str:
        """格式化单个 RetrievedChunk。"""
        source = (
            chunk.metadata.get("file_name")
            or chunk.metadata.get("source_path")
            or "unknown"
        )
        title = self._format_title(chunk)

        return "\n".join(
            [
                f"[资料 {index}]",
                f"来源: {source}",
                f"标题: {title}",
                "内容:",
                chunk.text,
            ]
        )

    def _format_title(self, chunk: RetrievedChunk) -> str:
        """从 metadata 中提取标题层级。"""
        header_path = chunk.metadata.get("header_path")
        if header_path:
            return str(header_path)

        headers = chunk.metadata.get("headers")
        if isinstance(headers, dict):
            title_parts = [
                headers.get("h1"),
                headers.get("h2"),
                headers.get("h3"),
            ]
            title_parts = [title for title in title_parts if title]

            if title_parts:
                return " > ".join(str(title) for title in title_parts)

        title_parts = [
            chunk.metadata.get("h1"),
            chunk.metadata.get("h2"),
            chunk.metadata.get("h3"),
        ]
        title_parts = [title for title in title_parts if title]

        if not title_parts:
            return "unknown"

        return " > ".join(str(title) for title in title_parts)
