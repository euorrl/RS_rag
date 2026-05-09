import os
from collections.abc import Iterator
from typing import Any

from dotenv import load_dotenv

from app.generator.generator_base import BaseGenerator, Prompt

OpenAI = None


def _get_openai_client_class():
    """懒加载 OpenAI SDK Client。"""
    global OpenAI

    if OpenAI is None:
        from openai import OpenAI as OpenAIClient

        OpenAI = OpenAIClient

    return OpenAI


class OpenAIGenerator(BaseGenerator):
    """OpenAI Generator。

    OpenAIGenerator 使用 OpenAI Responses API 调用 LLM，默认模型为
    "gpt-5.4-mini"。该类只负责执行 prompt -> LLM -> answer 的生成流程，
    不负责维护聊天记忆。

    默认使用流式接口；generate() 会在内部消费 stream_generate() 并拼接完整答案。
    """

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ):
        """初始化 OpenAI Generator。

        Args:
            model (str): 使用的 OpenAI 模型名称，默认 "gpt-5.4-mini"。
            api_key (str | None): OpenAI API Key。
                如果不传，则从环境变量 OPENAI_API_KEY 读取。
            base_url (str | None): 可选 OpenAI 兼容接口地址。
            client (Any | None): 已初始化的 OpenAI client。
                测试或高级用法中可以直接传入，避免重复初始化。
            request_kwargs (dict[str, Any] | None): 传递给 Responses API 的额外参数。

        Raises:
            TypeError:
                - model 不是 str
                - api_key 不是 str 或 None
                - base_url 不是 str 或 None
                - request_kwargs 不是 dict 或 None
            ValueError:
                - model 为空字符串
                - api_key 未传入且环境变量 OPENAI_API_KEY 未配置
        """
        self._validate_init_input(
            model=model,
            api_key=api_key,
            base_url=base_url,
            request_kwargs=request_kwargs,
        )

        self.model = model.strip()
        self.request_kwargs = request_kwargs or {}

        if client is not None:
            self.client = client
            return

        load_dotenv()

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未配置")

        client_kwargs = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        client_class = _get_openai_client_class()
        self.client = client_class(**client_kwargs)

    def generate(self, prompt: Prompt) -> str:
        """根据 prompt 生成完整答案。

        Args:
            prompt (Prompt): 输入给 LLM 的 prompt。
                可以是字符串 prompt，也可以是 chat messages。

        Returns:
            str: LLM 生成的完整答案。

        Raises:
            TypeError:
                - prompt 不是 str 或 list[dict[str, str]]
                - chat messages 格式不合法
            ValueError:
                - prompt 为空字符串
                - chat message role 不支持
        """
        return "".join(self.stream_generate(prompt))

    def stream_generate(self, prompt: Prompt) -> Iterator[str]:
        """根据 prompt 流式生成答案。

        Args:
            prompt (Prompt): 输入给 LLM 的 prompt。
                可以是字符串 prompt，也可以是 chat messages。

        Yields:
            str: LLM 流式返回的文本片段。

        Raises:
            TypeError:
                - prompt 不是 str 或 list[dict[str, str]]
                - chat messages 格式不合法
            ValueError:
                - prompt 为空字符串
                - chat message role 不支持
        """
        response_input = self._normalize_prompt(prompt)
        stream = self.client.responses.create(
            model=self.model,
            input=response_input,
            stream=True,
            **self.request_kwargs,
        )

        for event in stream:
            text_delta = self._extract_text_delta(event)
            if text_delta:
                yield text_delta

    def _validate_init_input(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None,
        request_kwargs: dict[str, Any] | None,
    ) -> None:
        """校验初始化参数。"""
        if not isinstance(model, str):
            raise TypeError("model 必须是 str 类型")

        if not model.strip():
            raise ValueError("model 不能为空字符串")

        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("api_key 必须是 str 或 None")

        if base_url is not None and not isinstance(base_url, str):
            raise TypeError("base_url 必须是 str 或 None")

        if request_kwargs is not None and not isinstance(request_kwargs, dict):
            raise TypeError("request_kwargs 必须是 dict 或 None")

    def _normalize_prompt(self, prompt: Prompt) -> Prompt:
        """规范化 prompt 输入。"""
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("prompt 不能为空字符串")

            return prompt.strip()

        if not isinstance(prompt, list):
            raise TypeError("prompt 必须是 str 或 list[dict[str, str]]")

        normalized_messages = []
        for message in prompt:
            normalized_messages.append(self._normalize_message(message))

        if not normalized_messages:
            raise ValueError("prompt messages 不能为空列表")

        return normalized_messages

    def _normalize_message(self, message: dict[str, str]) -> dict[str, str]:
        """规范化单条 chat message。"""
        if not isinstance(message, dict):
            raise TypeError("prompt messages 中的元素必须是 dict")

        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str):
            raise TypeError("message role 必须是 str 类型")

        if not isinstance(content, str):
            raise TypeError("message content 必须是 str 类型")

        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")

        if not content.strip():
            raise ValueError("message content 不能为空字符串")

        return {
            "role": role,
            "content": content.strip(),
        }

    def _extract_text_delta(self, event: Any) -> str | None:
        """从 OpenAI stream event 中提取文本增量。"""
        event_type = self._get_value(event, "type")

        if event_type == "response.output_text.delta":
            delta = self._get_value(event, "delta")
            if isinstance(delta, str):
                return delta

        choices = self._get_value(event, "choices")
        if choices:
            return self._extract_chat_completion_delta(choices)

        return None

    def _extract_chat_completion_delta(self, choices: Any) -> str | None:
        """兼容 chat completions 风格的流式 chunk。"""
        first_choice = choices[0]
        delta = self._get_value(first_choice, "delta")
        content = self._get_value(delta, "content")

        if isinstance(content, str):
            return content

        return None

    def _get_value(self, obj: Any, key: str) -> Any:
        """同时兼容 dict 和对象属性读取。"""
        if isinstance(obj, dict):
            return obj.get(key)

        return getattr(obj, key, None)
