import pytest

from app.prompt_builder.chat_messages_prompt_builder import ChatMessagesPromptBuilder
from app.prompt_builder.prompt_builder_factory import (
    build_messages_prompt,
    build_prompt,
    build_string_prompt,
    get_prompt_builder,
)
from app.prompt_builder.string_prompt_builder import StringPromptBuilder
from app.schemas import RetrievedChunk

pytestmark = pytest.mark.prompt_builder


class FakeMessagesPromptBuilder:
    """模拟 messages PromptBuilder，避免 factory 测试依赖真实模板细节。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self, query, retrieved_chunks, history=None):
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"{query}:{len(retrieved_chunks)}"},
        ]


class FakeStringPromptBuilder:
    """模拟 string PromptBuilder，避免 factory 测试依赖真实模板细节。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self, query, retrieved_chunks, history=None):
        return f"{query}:{len(retrieved_chunks)}"


def make_retrieved_chunks() -> list[RetrievedChunk]:
    """创建测试用 RetrievedChunk 列表。"""
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="candidate",
            score=0.9,
        )
    ]


def test_get_prompt_builder_returns_chat_messages_prompt_builder_by_default():
    """验证 get_prompt_builder 默认返回 ChatMessagesPromptBuilder。"""
    prompt_builder = get_prompt_builder()

    assert isinstance(prompt_builder, ChatMessagesPromptBuilder)


def test_get_prompt_builder_supports_chat_messages_alias():
    """验证 chat 和 chat_messages 都会返回 ChatMessagesPromptBuilder。"""
    assert isinstance(get_prompt_builder(provider="CHAT"), ChatMessagesPromptBuilder)
    assert isinstance(
        get_prompt_builder(provider="chat_messages"),
        ChatMessagesPromptBuilder,
    )


def test_get_prompt_builder_returns_string_prompt_builder():
    """验证 provider 为 string 时返回 StringPromptBuilder。"""
    prompt_builder = get_prompt_builder(provider="string")

    assert isinstance(prompt_builder, StringPromptBuilder)


def test_get_prompt_builder_passes_kwargs(monkeypatch):
    """验证 get_prompt_builder 会透传初始化参数给具体 PromptBuilder。"""
    monkeypatch.setattr(
        "app.prompt_builder.prompt_builder_factory.ChatMessagesPromptBuilder",
        FakeMessagesPromptBuilder,
    )

    prompt_builder = get_prompt_builder(
        provider="chat",
        system_prompt="custom",
        max_context_chars=100,
        allow_general_fallback=False,
    )

    assert isinstance(prompt_builder, FakeMessagesPromptBuilder)
    assert prompt_builder.kwargs == {
        "system_prompt": "custom",
        "max_context_chars": 100,
        "allow_general_fallback": False,
    }


def test_get_prompt_builder_rejects_invalid_provider():
    """验证 get_prompt_builder 会拒绝非法 provider 类型和不支持的 provider。"""
    with pytest.raises(TypeError):
        get_prompt_builder(provider=123)

    with pytest.raises(ValueError):
        get_prompt_builder(provider="unknown")


def test_build_prompt_convenience_entrypoint_defaults_to_messages(monkeypatch):
    """验证 build_prompt 默认会创建 chat messages prompt。"""
    monkeypatch.setattr(
        "app.prompt_builder.prompt_builder_factory.ChatMessagesPromptBuilder",
        FakeMessagesPromptBuilder,
    )

    prompt = build_prompt(
        query="query",
        retrieved_chunks=make_retrieved_chunks(),
        system_prompt="custom",
    )

    assert prompt == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "query:1"},
    ]


def test_build_messages_prompt_convenience_entrypoint(monkeypatch):
    """验证 build_messages_prompt 会返回 chat messages。"""
    monkeypatch.setattr(
        "app.prompt_builder.prompt_builder_factory.ChatMessagesPromptBuilder",
        FakeMessagesPromptBuilder,
    )

    prompt = build_messages_prompt(
        query="query",
        retrieved_chunks=make_retrieved_chunks(),
        history=[{"role": "user", "content": "history"}],
    )

    assert isinstance(prompt, list)
    assert prompt[1]["content"] == "query:1"


def test_build_string_prompt_convenience_entrypoint(monkeypatch):
    """验证 build_string_prompt 会返回字符串 prompt。"""
    monkeypatch.setattr(
        "app.prompt_builder.prompt_builder_factory.StringPromptBuilder",
        FakeStringPromptBuilder,
    )

    prompt = build_string_prompt(
        query="query",
        retrieved_chunks=make_retrieved_chunks(),
        system_prompt="custom",
        allow_general_fallback=False,
    )

    assert prompt == "query:1"


def test_typed_prompt_helpers_reject_wrong_return_type(monkeypatch):
    """验证明确返回类型的便捷入口会拒绝错误 provider 返回值。"""
    monkeypatch.setattr(
        "app.prompt_builder.prompt_builder_factory.ChatMessagesPromptBuilder",
        FakeMessagesPromptBuilder,
    )
    monkeypatch.setattr(
        "app.prompt_builder.prompt_builder_factory.StringPromptBuilder",
        FakeStringPromptBuilder,
    )

    with pytest.raises(TypeError):
        build_messages_prompt("query", make_retrieved_chunks(), provider="string")

    with pytest.raises(TypeError):
        build_string_prompt("query", make_retrieved_chunks(), provider="chat")
