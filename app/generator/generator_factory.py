from collections.abc import Iterator

from app.generator.generator_base import BaseGenerator, Prompt
from app.generator.openai_generator import OpenAIGenerator


def get_generator(
    provider: str = "openai",
    **kwargs,
) -> BaseGenerator:
    """根据 provider 获取对应的 Generator 实例。

    Args:
        provider (str): Generator 类型。
            当前支持：
            - "openai"
        **kwargs: 传递给具体 Generator 的初始化参数，例如：
            - model (str): 使用的模型名称
            - api_key (str | None): OpenAI API Key
            - base_url (str | None): OpenAI 兼容接口地址
            - request_kwargs (dict | None): 传递给模型接口的额外参数

    Returns:
        BaseGenerator: 对应的 Generator 实例。

    Raises:
        TypeError: provider 不是字符串时抛出。
        ValueError: provider 不支持时抛出。
    """
    if not isinstance(provider, str):
        raise TypeError("provider 必须是 str 类型")

    provider = provider.lower()

    if provider == "openai":
        return OpenAIGenerator(**kwargs)

    raise ValueError(f"Unsupported generator provider: {provider}")


def generate(
    prompt: Prompt,
    provider: str = "openai",
    **kwargs,
) -> str:
    """根据 prompt 调用 LLM 生成完整答案。

    该函数是 Generator 层的轻量封装，用于简化调用流程。
    1. 根据 provider 选择具体 Generator。
    2. 将 prompt 传递给 Generator.generate()。
    3. 返回完整答案文本。

    Args:
        prompt (Prompt): 输入给 LLM 的 prompt。
            可以是字符串 prompt，也可以是 chat messages。
        provider (str): Generator 类型，默认 "openai"。
        **kwargs: 传递给具体 Generator 的初始化参数，例如：
            - model (str): 使用的模型名称
            - api_key (str | None): OpenAI API Key
            - base_url (str | None): OpenAI 兼容接口地址
            - request_kwargs (dict | None): 传递给模型接口的额外参数

    Returns:
        str: LLM 生成的完整答案。

    Raises:
        TypeError:
            - prompt 格式不合法
            - provider 不是 str
        ValueError:
            - prompt 为空
            - provider 不支持
    """
    generator = get_generator(provider=provider, **kwargs)
    return generator.generate(prompt=prompt)


def stream_generate(
    prompt: Prompt,
    provider: str = "openai",
    **kwargs,
) -> Iterator[str]:
    """根据 prompt 调用 LLM 流式生成答案。

    Args:
        prompt (Prompt): 输入给 LLM 的 prompt。
            可以是字符串 prompt，也可以是 chat messages。
        provider (str): Generator 类型，默认 "openai"。
        **kwargs: 传递给具体 Generator 的初始化参数，例如：
            - model (str): 使用的模型名称
            - api_key (str | None): OpenAI API Key
            - base_url (str | None): OpenAI 兼容接口地址
            - request_kwargs (dict | None): 传递给模型接口的额外参数

    Yields:
        str: LLM 流式返回的文本片段。

    Raises:
        TypeError:
            - prompt 格式不合法
            - provider 不是 str
        ValueError:
            - prompt 为空
            - provider 不支持
    """
    generator = get_generator(provider=provider, **kwargs)
    yield from generator.stream_generate(prompt=prompt)
