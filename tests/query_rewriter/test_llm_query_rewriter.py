import pytest

from app.query_rewriter import LLMQueryRewriter

pytestmark = pytest.mark.query_rewriter


class FakeGenerator:
    """模拟 Generator，避免测试时调用真实大模型。"""

    def __init__(self, response: str = "改写后的问题"):
        self.response = response
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.response


def make_history() -> list[dict[str, str]]:
    """构造测试用历史对话。"""
    return [
        {
            "role": "user",
            "content": "什么是 NDVI?",
        },
        {
            "role": "assistant",
            "content": "NDVI 是归一化植被指数。",
        },
    ]


def test_rewrite_returns_original_query_without_history():
    """验证无 history 时直接返回去除首尾空白后的原始 query。"""
    generator = FakeGenerator()
    rewriter = LLMQueryRewriter(generator=generator)

    result = rewriter.rewrite("  它有什么用途？  ", history=None)

    assert result == "它有什么用途？"
    assert generator.calls == []


def test_rewrite_calls_generator_with_history():
    """验证有 history 时会构建 messages 并调用 generator。"""
    generator = FakeGenerator(response="NDVI 有什么用途？")
    rewriter = LLMQueryRewriter(generator=generator)

    result = rewriter.rewrite(
        query="它有什么用途？",
        history=make_history(),
    )

    assert result == "NDVI 有什么用途？"
    assert len(generator.calls) == 1
    messages = generator.calls[0]
    assert messages[0]["role"] == "system"
    assert "问题改写器" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "用户：什么是 NDVI?" in messages[1]["content"]
    assert "助手：NDVI 是归一化植被指数。" in messages[1]["content"]
    assert "当前问题：\n它有什么用途？" in messages[1]["content"]


def test_rewrite_returns_generator_response():
    """验证 rewrite() 会返回大模型输出的改写问题。"""
    generator = FakeGenerator(response="NDVI 在遥感中有什么用途？")
    rewriter = LLMQueryRewriter(generator=generator)

    result = rewriter.rewrite(
        query="有什么用途？",
        history=make_history(),
    )

    assert result == "NDVI 在遥感中有什么用途？"


def test_rewrite_fallbacks_to_original_query_when_generator_returns_empty():
    """验证大模型输出空字符串时 fallback 返回原始 query。"""
    generator = FakeGenerator(response="   ")
    rewriter = LLMQueryRewriter(generator=generator)

    result = rewriter.rewrite(
        query=" 它有什么用途？ ",
        history=make_history(),
    )

    assert result == "它有什么用途？"


def test_rewrite_rejects_non_string_generator_response():
    """验证大模型输出非字符串时会抛出类型错误。"""
    generator = FakeGenerator(response=123)
    rewriter = LLMQueryRewriter(generator=generator)

    with pytest.raises(TypeError):
        rewriter.rewrite(
            query="它有什么用途？",
            history=make_history(),
        )


def test_rewrite_rejects_invalid_query():
    """验证 rewrite() 会拒绝非字符串或空字符串 query。"""
    rewriter = LLMQueryRewriter(generator=FakeGenerator())

    with pytest.raises(TypeError):
        rewriter.rewrite(query=123, history=None)

    with pytest.raises(ValueError):
        rewriter.rewrite(query="   ", history=None)


def test_format_history_formats_user_and_assistant_messages():
    """验证 _format_history() 会按用户和助手角色格式化历史消息。"""
    rewriter = LLMQueryRewriter(generator=FakeGenerator())

    result = rewriter._format_history(make_history())

    assert result == "用户：什么是 NDVI?\n助手：NDVI 是归一化植被指数。"


def test_format_history_rejects_invalid_history():
    """验证 _format_history() 会拒绝非法 history 结构。"""
    rewriter = LLMQueryRewriter(generator=FakeGenerator())

    with pytest.raises(TypeError):
        rewriter._format_history("history")

    with pytest.raises(TypeError):
        rewriter._format_history(["message"])

    with pytest.raises(TypeError):
        rewriter._format_history([{"role": 123, "content": "content"}])

    with pytest.raises(TypeError):
        rewriter._format_history([{"role": "user", "content": 123}])

    with pytest.raises(ValueError):
        rewriter._format_history([{"role": "system", "content": "system"}])


def test_init_rejects_generator_without_generate_method():
    """验证初始化时会拒绝不具备 generate() 方法的对象。"""
    with pytest.raises(TypeError):
        LLMQueryRewriter(generator=object())
