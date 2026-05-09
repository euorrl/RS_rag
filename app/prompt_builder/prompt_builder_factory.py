from app.prompt_builder.chat_messages_prompt_builder import ChatMessagesPromptBuilder
from app.prompt_builder.prompt_builder_base import BasePromptBuilder, Prompt
from app.prompt_builder.string_prompt_builder import StringPromptBuilder
from app.schemas import RetrievedChunk


def get_prompt_builder(
    provider: str = "chat",
    **kwargs,
) -> BasePromptBuilder:
    """根据 provider 获取对应的 PromptBuilder 实例。

    Args:
        provider (str): PromptBuilder 类型。
            当前支持：
            - "chat"
            - "string"
        **kwargs: 传递给具体 PromptBuilder 的初始化参数，例如：
            - system_prompt (str): 自定义系统提示词
            - max_context_chars (int | None): 参考资料部分的最大字符数
            - allow_general_fallback (bool): 资料不足时是否允许通用知识补充

    Returns:
        BasePromptBuilder: 对应的 PromptBuilder 实例。

    Raises:
        TypeError: provider 不是字符串时抛出。
        ValueError: provider 不支持时抛出。
    """
    if not isinstance(provider, str):
        raise TypeError("provider 必须是 str 类型")

    provider = provider.lower()

    if provider in {"chat", "chat_messages"}:
        return ChatMessagesPromptBuilder(**kwargs)

    if provider == "string":
        return StringPromptBuilder(**kwargs)

    raise ValueError(f"Unsupported prompt builder provider: {provider}")


def build_prompt(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    provider: str = "chat",
    history: list[dict[str, str]] | None = None,
    **kwargs,
) -> Prompt:
    """根据 query 和 RetrievedChunk 列表构建 prompt。

    该函数是 PromptBuilder 层的通用轻量封装，用于简化调用流程。
    默认返回 chat messages；如果需要明确的返回类型，优先使用
    build_messages_prompt() 或 build_string_prompt()。

    Args:
        query (str): 用户查询文本。
        retrieved_chunks (list[RetrievedChunk]): 已召回或已重排的候选 chunks。
        provider (str): PromptBuilder 类型，默认 "chat"。
        history (list[dict[str, str]] | None): 可选历史对话消息。
            仅 chat messages builder 使用。
        **kwargs: 传递给具体 PromptBuilder 的初始化参数，例如：
            - system_prompt (str): 自定义系统提示词
            - max_context_chars (int | None): 参考资料部分的最大字符数
            - allow_general_fallback (bool): 资料不足时是否允许通用知识补充

    Returns:
        Prompt: 构建完成的 prompt，可以是字符串或 chat messages。

    Raises:
        TypeError:
            - query 不是 str
            - retrieved_chunks 不是 list[RetrievedChunk]
            - provider 不是 str
        ValueError:
            - query 为空字符串
            - provider 不支持
    """
    prompt_builder = get_prompt_builder(provider=provider, **kwargs)
    return prompt_builder.build(
        query=query,
        retrieved_chunks=retrieved_chunks,
        history=history,
    )


def build_messages_prompt(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    provider: str = "chat",
    history: list[dict[str, str]] | None = None,
    **kwargs,
) -> list[dict[str, str]]:
    """根据 query 和 RetrievedChunk 列表构建 chat messages prompt。

    Args:
        query (str): 用户查询文本。
        retrieved_chunks (list[RetrievedChunk]): 已召回或已重排的候选 chunks。
        provider (str): PromptBuilder 类型，默认 "chat"。
        history (list[dict[str, str]] | None): 可选历史对话消息。
        **kwargs: 传递给具体 PromptBuilder 的初始化参数。

    Returns:
        list[dict[str, str]]: 可直接传递给 Chat API 的 messages。

    Raises:
        TypeError: provider 对应的 PromptBuilder 未返回 chat messages 时抛出。
    """
    prompt = build_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        provider=provider,
        history=history,
        **kwargs,
    )

    if not isinstance(prompt, list):
        raise TypeError("provider 对应的 PromptBuilder 未返回 chat messages")

    return prompt


def build_string_prompt(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    provider: str = "string",
    **kwargs,
) -> str:
    """根据 query 和 RetrievedChunk 列表构建字符串 prompt。

    Args:
        query (str): 用户查询文本。
        retrieved_chunks (list[RetrievedChunk]): 已召回或已重排的候选 chunks。
        provider (str): PromptBuilder 类型，默认 "string"。
        **kwargs: 传递给具体 PromptBuilder 的初始化参数。

    Returns:
        str: 可直接传递给 completion-style 模型的字符串 prompt。

    Raises:
        TypeError: provider 对应的 PromptBuilder 未返回字符串时抛出。
    """
    prompt = build_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        provider=provider,
        history=None,
        **kwargs,
    )

    if not isinstance(prompt, str):
        raise TypeError("provider 对应的 PromptBuilder 未返回字符串 prompt")

    return prompt
