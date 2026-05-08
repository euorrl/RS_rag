from app.prompt_builder.string_prompt_builder import StringPromptBuilder
from app.schemas import RetrievedChunk


class ChatMessagesPromptBuilder(StringPromptBuilder):
    """Chat messages PromptBuilder。

    ChatMessagesPromptBuilder 用于生成 chat messages 格式 prompt，适合 Chat API、
    多轮问答和保存历史聊天记录。返回格式为：
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ]
    """

    allowed_history_roles = {"user", "assistant", "system"}

    def build(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """根据 query、RetrievedChunk 和可选 history 构建 chat messages。

        Args:
            query (str): 用户查询文本。
            retrieved_chunks (list[RetrievedChunk]): 已召回或已重排的候选 chunks。
            history (list[dict[str, str]] | None): 可选历史对话消息。
                只保留 role 为 user 或 assistant 的历史消息，过滤 system 历史消息，
                避免出现多个 system prompt。

        Returns:
            list[dict[str, str]]: 可直接传递给 Chat API 的 messages。

        Raises:
            TypeError:
                - query 不是 str
                - retrieved_chunks 不是 list
                - retrieved_chunks 中元素不是 RetrievedChunk
                - history 不是 list 或 None
                - history 中元素不是 dict[str, str]
            ValueError:
                - query 为空字符串
                - history 中存在不支持的 role
        """
        self._validate_build_input(query, retrieved_chunks)
        history_messages = self._normalize_history(history)

        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            *history_messages,
            {
                "role": "user",
                "content": self._build_user_prompt(query, retrieved_chunks),
            },
        ]

    def _normalize_history(
        self,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """校验并规范化历史消息。"""
        if history is None:
            return []

        if not isinstance(history, list):
            raise TypeError("history 必须是 list 或 None")

        normalized_history = []
        for message in history:
            if not isinstance(message, dict):
                raise TypeError("history 中的元素必须是 dict")

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str):
                raise TypeError("history 中每条消息的 role 必须是 str")

            if not isinstance(content, str):
                raise TypeError("history 中每条消息的 content 必须是 str")

            if role not in self.allowed_history_roles:
                raise ValueError(f"Unsupported history role: {role}")

            if role == "system":
                continue

            normalized_history.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return normalized_history
