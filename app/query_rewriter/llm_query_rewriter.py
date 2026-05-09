from app.generator.generator_base import BaseGenerator


class LLMQueryRewriter:
    """基于 LLM 的查询改写器。

    LLMQueryRewriter 用于多轮 RAG 问答中，根据历史对话和当前用户问题，
    将当前问题改写成独立、明确、适合检索的问题。

    该类只负责问题改写，不负责：
    - 回答用户问题
    - 召回候选 chunks
    - 重排候选 chunks
    - 构建最终问答 prompt
    - 保存或维护 memory

    注意：
    LLMQueryRewriter 不直接调用 OpenAI SDK，而是复用外部传入的 generator.generate()。
    """

    allowed_history_roles = {"user", "assistant"}

    def __init__(self, generator: BaseGenerator):
        """初始化 LLMQueryRewriter。

        Args:
            generator (BaseGenerator): 已初始化的 Generator 实例。
                rewrite() 会调用 generator.generate(messages) 获取改写后的检索问题。

        Raises:
            TypeError: generator 不具备 generate() 方法时抛出。
        """
        if not hasattr(generator, "generate"):
            raise TypeError("generator 必须提供 generate() 方法")

        self.generator = generator

    def rewrite(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """根据历史对话改写当前检索问题。

        处理流程：
        1. 校验 query 必须是非空字符串。
        2. 如果 history 为空，则直接返回 query.strip()。
        3. 如果 history 不为空，则构建 chat messages 调用 generator。
        4. 如果模型输出为空，则 fallback 返回原始 query.strip()。

        Args:
            query (str): 当前用户问题。
            history (list[dict[str, str]] | None): 历史对话 messages。
                格式通常为：
                [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                ]

        Returns:
            str: 改写后的检索问题。

        Raises:
            TypeError:
                - query 不是 str
                - history 不是 list 或 None
                - history 中元素不是 dict
                - history 中 role 或 content 不是 str
                - generator 返回结果不是 str
            ValueError:
                - query 为空字符串
                - history 中存在不支持的 role
        """
        query = self._validate_query(query)

        if not history:
            return query

        formatted_history = self._format_history(history)
        messages = self._build_rewrite_messages(
            query=query,
            formatted_history=formatted_history,
        )
        rewritten_query = self.generator.generate(messages)

        if not isinstance(rewritten_query, str):
            raise TypeError("query rewrite 结果必须是 str 类型")

        rewritten_query = rewritten_query.strip()
        if not rewritten_query:
            return query

        return rewritten_query

    def _format_history(self, history: list[dict[str, str]]) -> str:
        """将 history messages 格式化为适合问题改写的文本。

        Args:
            history (list[dict[str, str]]): 历史对话 messages。

        Returns:
            str: 格式化后的历史对话文本。

        Raises:
            TypeError:
                - history 不是 list
                - history 中元素不是 dict
                - history 中 role 或 content 不是 str
            ValueError: history 中存在不支持的 role。
        """
        if not isinstance(history, list):
            raise TypeError("history 必须是 list 类型")

        formatted_lines = []
        for message in history:
            if not isinstance(message, dict):
                raise TypeError("history 中的元素必须是 dict 类型")

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str):
                raise TypeError("history 中每条消息的 role 必须是 str 类型")

            if not isinstance(content, str):
                raise TypeError("history 中每条消息的 content 必须是 str 类型")

            if role not in self.allowed_history_roles:
                raise ValueError(f"Unsupported history role: {role}")

            role_name = "用户" if role == "user" else "助手"
            formatted_lines.append(f"{role_name}：{content.strip()}")

        return "\n".join(formatted_lines)

    def _validate_query(self, query: str) -> str:
        """校验并规范化 query。"""
        if not isinstance(query, str):
            raise TypeError("query 必须是 str 类型")

        query = query.strip()
        if not query:
            raise ValueError("query 不能为空字符串")

        return query

    def _build_rewrite_messages(
        self,
        query: str,
        formatted_history: str,
    ) -> list[dict[str, str]]:
        """构建问题改写用 chat messages。"""
        return [
            {
                "role": "system",
                "content": (
                    "你是 RAG 系统中的问题改写器。\n"
                    "你的任务是把当前问题改写成独立、明确、适合检索的问题。\n"
                    "只输出改写后的问题，不要回答问题。\n"
                    "如果当前问题已经完整，则原样返回。\n"
                    "不要添加历史中没有的新概念。\n"
                    "保留原问题语言。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "历史对话：\n"
                    f"{formatted_history}\n\n"
                    "当前问题：\n"
                    f"{query}\n\n"
                    "请输出改写后的检索问题。"
                ),
            },
        ]
