from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """一轮完整历史对话（ConversationTurn）。

    ConversationTurn 用于保存真实发生过的一轮 user-assistant 对话。

    注意：
    - 不保存 system prompt；
    - 不保存 RAG 参考资料；
    - 只保存用户原始问题和助手最终回答；
    - 当前轮还没有生成回答时，不应该写入 ConversationTurn。
    """

    user_content: str
    """用户原始问题。"""

    assistant_content: str
    """助手最终回答。"""

    def __post_init__(self) -> None:
        """校验单轮对话内容。"""
        if not isinstance(self.user_content, str):
            raise TypeError("user_content 必须是 str 类型")

        if not isinstance(self.assistant_content, str):
            raise TypeError("assistant_content 必须是 str 类型")

        if not self.user_content.strip():
            raise ValueError("user_content 不能为空字符串")

        if not self.assistant_content.strip():
            raise ValueError("assistant_content 不能为空字符串")

        self.user_content = self.user_content.strip()
        self.assistant_content = self.assistant_content.strip()

    def to_messages(self) -> list[dict[str, str]]:
        """转换为 Chat API history messages 格式。

        Returns:
            list[dict[str, str]]: 一轮对话展开后的 user / assistant messages。
        """
        return [
            {
                "role": "user",
                "content": self.user_content,
            },
            {
                "role": "assistant",
                "content": self.assistant_content,
            },
        ]


@dataclass
class ChatHistory:
    """聊天历史状态（ChatHistory）。

    ChatHistory 用于保存多轮对话的历史状态。

    它不是最终发送给 LLM 的完整 messages。
    最终 messages 应该由 PromptBuilder 临时构建：

    system prompt
    + ChatHistory.to_messages()
    + 当前 RAG user message

    包含：
    - summary：较早历史对话摘要，可选；
    - recent_turns：最近几轮完整 user-assistant 对话。
    """

    summary: str | None = None
    """较早历史对话的摘要。"""

    recent_turns: list[ConversationTurn] = field(default_factory=list)
    """最近几轮完整对话。"""

    def __post_init__(self) -> None:
        """校验聊天历史字段。"""
        if self.summary is not None:
            if not isinstance(self.summary, str):
                raise TypeError("summary 必须是 str 或 None")

            self.summary = self.summary.strip() or None

        if not isinstance(self.recent_turns, list):
            raise TypeError("recent_turns 必须是 list 类型")

        if not all(isinstance(turn, ConversationTurn) for turn in self.recent_turns):
            raise TypeError("recent_turns 中的元素必须是 ConversationTurn")

    def add_turn(
        self,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """添加一轮完整对话。"""
        self.recent_turns.append(
            ConversationTurn(
                user_content=user_content,
                assistant_content=assistant_content,
            )
        )

    def to_messages(self) -> list[dict[str, str]]:
        """转换为可传给 PromptBuilder 的 history messages。

        返回结果不包含 system prompt。

        如果存在 summary，则将 summary 转换成一条 assistant 消息，
        放在最近对话之前。
        """
        messages: list[dict[str, str]] = []

        if self.summary:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"以下是较早对话的摘要：\n{self.summary}",
                }
            )

        for turn in self.recent_turns:
            messages.extend(turn.to_messages())

        return messages

    def clear(self) -> None:
        """清空聊天历史。"""
        self.summary = None
        self.recent_turns.clear()

    def has_history(self) -> bool:
        """判断是否存在历史摘要或最近对话。"""
        return bool(self.summary or self.recent_turns)
