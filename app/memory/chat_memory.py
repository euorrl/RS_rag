from app.generator import generate
from app.schemas import ChatHistory, ConversationTurn


class ChatMemory:
    """轻量级聊天记忆。

    ChatMemory 用于多轮 RAG 问答中的历史对话管理，内部使用 ChatHistory schema
    保存较早对话摘要和最近若干轮完整对话。

    ChatMemory 只负责保存历史对话，不负责：
    - prompt 构建
    - 召回
    - 重排
    - 问题重写

    注意：
    - 不保存 system prompt；
    - 不保存 RAG 参考资料；
    - 只保存用户原始问题和助手最终回答；
    - 当前轮 assistant 生成完成后，pipeline 才应该调用 add_turn()。
    """

    def __init__(self, max_turns: int = 5):
        """初始化 ChatMemory。

        Args:
            max_turns (int): 最多保留最近 max_turns 轮完整问答。
                一轮问答对应一个 ConversationTurn。

        Raises:
            TypeError: max_turns 不是 int 时抛出。
            ValueError: max_turns 不大于 0 时抛出。
        """
        if not isinstance(max_turns, int):
            raise TypeError("max_turns 必须是 int 类型")

        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0")

        self.max_turns = max_turns
        self.history = ChatHistory()

    def add_turn(
        self,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """添加一轮完整对话。

        添加后如果 recent_turns 超过 max_turns，则将最早一轮对话和已有 summary
        重新总结压缩到 ChatHistory.summary 中，只保留最近 max_turns 轮完整对话。

        Args:
            user_content (str): 用户原始问题。
            assistant_content (str): 助手最终回答。

        Raises:
            TypeError:
                - user_content 不是 str
                - assistant_content 不是 str
            ValueError:
                - user_content 为空字符串
                - assistant_content 为空字符串
        """
        self.history.add_turn(
            user_content=user_content,
            assistant_content=assistant_content,
        )
        self._compress_overflow_turns()

    def get_messages(self) -> list[dict[str, str]]:
        """返回可传给 ChatMessagesPromptBuilder 的 history messages。

        返回结果不包含 system prompt。

        Returns:
            list[dict[str, str]]: 历史对话 messages。
        """
        return self.history.to_messages()

    def clear(self) -> None:
        """清空聊天记忆。"""
        self.history.clear()

    def has_history(self) -> bool:
        """判断是否存在历史摘要或最近对话。"""
        return self.history.has_history()

    def get_history(self) -> ChatHistory:
        """返回内部 ChatHistory 对象，方便调试或后续扩展。"""
        return self.history

    def _compress_overflow_turns(self) -> None:
        """将超过窗口的早期对话压缩到 summary 中。"""
        while len(self.history.recent_turns) > self.max_turns:
            overflow_turn = self.history.recent_turns.pop(0)
            self.history.summary = self._summarize_overflow_turn(overflow_turn)

    def _summarize_overflow_turn(self, overflow_turn: ConversationTurn) -> str:
        """调用 LLM 将已有 summary 与溢出的最早一轮对话合并成新摘要。"""
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一个对话历史摘要助手。请将已有摘要和新加入的早期对话"
                    "合并成一段准确、保留关键信息的中文摘要。"
                    "摘要只应包含用户问题和助手最终回答中的长期有用信息，"
                    "不要加入 system prompt、RAG 参考资料或不存在的内容。"
                ),
            },
            {
                "role": "user",
                "content": self._build_summary_prompt(overflow_turn),
            },
        ]
        summary = generate(
            prompt=prompt,
            provider="openai",
            model="gpt-5.4-mini",
        )

        if not isinstance(summary, str):
            raise TypeError("summary 生成结果必须是 str 类型")

        summary = summary.strip()
        if not summary:
            raise ValueError("summary 生成结果不能为空字符串")

        return summary

    def _build_summary_prompt(self, overflow_turn: ConversationTurn) -> str:
        """构建 summary 压缩 prompt。"""
        existing_summary = self.history.summary or "无"

        return "\n\n".join(
            [
                "已有摘要：",
                existing_summary,
                "需要合并进摘要的早期对话：",
                f"用户：{overflow_turn.user_content}",
                f"助手：{overflow_turn.assistant_content}",
                "请输出更新后的摘要。",
            ]
        )
