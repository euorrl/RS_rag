import pytest

from app.generator.generator_factory import generate, get_generator, stream_generate
from app.generator.openai_generator import OpenAIGenerator

pytestmark = pytest.mark.generator


class FakeGenerator:
    """模拟 Generator，避免 factory 测试调用真实模型。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompt):
        return f"answer:{prompt}"

    def stream_generate(self, prompt):
        yield "answer"
        yield f":{prompt}"


def test_get_generator_returns_openai_generator():
    """验证 get_generator 默认返回 OpenAIGenerator。"""
    generator = get_generator(client=object())

    assert isinstance(generator, OpenAIGenerator)


def test_get_generator_provider_case_insensitive(monkeypatch):
    """验证 get_generator 的 provider 参数大小写不敏感。"""
    monkeypatch.setattr(
        "app.generator.generator_factory.OpenAIGenerator",
        FakeGenerator,
    )

    generator = get_generator(
        provider="OPENAI",
        model="fake-model",
        request_kwargs={"max_output_tokens": 64},
    )

    assert isinstance(generator, FakeGenerator)
    assert generator.kwargs == {
        "model": "fake-model",
        "request_kwargs": {"max_output_tokens": 64},
    }


def test_get_generator_rejects_invalid_provider():
    """验证 get_generator 会拒绝非法 provider 类型和不支持的 provider。"""
    with pytest.raises(TypeError):
        get_generator(provider=123)

    with pytest.raises(ValueError):
        get_generator(provider="unknown")


def test_generate_convenience_entrypoint(monkeypatch):
    """验证 generate 便捷入口会创建 Generator 并返回完整答案。"""
    monkeypatch.setattr(
        "app.generator.generator_factory.OpenAIGenerator",
        FakeGenerator,
    )

    answer = generate(
        prompt="prompt",
        provider="openai",
        model="fake-model",
    )

    assert answer == "answer:prompt"


def test_stream_generate_convenience_entrypoint(monkeypatch):
    """验证 stream_generate 便捷入口会创建 Generator 并流式返回答案。"""
    monkeypatch.setattr(
        "app.generator.generator_factory.OpenAIGenerator",
        FakeGenerator,
    )

    chunks = list(
        stream_generate(
            prompt="prompt",
            provider="openai",
            model="fake-model",
        )
    )

    assert chunks == ["answer", ":prompt"]
