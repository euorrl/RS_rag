import pytest

from app.generator import openai_generator
from app.generator.openai_generator import OpenAIGenerator

pytestmark = pytest.mark.generator


class FakeEvent:
    """模拟 OpenAI Responses API 的流式事件对象。"""

    def __init__(self, event_type, delta=None):
        self.type = event_type
        self.delta = delta


class FakeDelta:
    """模拟 chat completions delta 对象。"""

    def __init__(self, content):
        self.content = content


class FakeChoice:
    """模拟 chat completions choice 对象。"""

    def __init__(self, content):
        self.delta = FakeDelta(content)


class FakeResponses:
    """模拟 OpenAI client.responses。"""

    def __init__(self):
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return [
            FakeEvent("response.output_text.delta", "你"),
            {"type": "response.output_text.delta", "delta": "好"},
            {"choices": [FakeChoice("！")]},
            FakeEvent("response.completed"),
        ]


class FakeClient:
    """模拟 OpenAI client。"""

    def __init__(self):
        self.responses = FakeResponses()


class FakeOpenAI:
    """模拟 OpenAI SDK Client。"""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = FakeResponses()
        self.instances.append(self)


def test_openai_generator_initializes_client_from_env(monkeypatch):
    """验证 OpenAIGenerator 会从环境变量初始化 OpenAI client。"""
    FakeOpenAI.instances = []
    monkeypatch.setattr(openai_generator, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    generator = OpenAIGenerator(base_url="https://example.com/v1")

    assert generator.model == "gpt-5.4-mini"
    assert FakeOpenAI.instances[0].kwargs == {
        "api_key": "test-key",
        "base_url": "https://example.com/v1",
    }


def test_openai_generator_uses_existing_client_and_streams_answer():
    """验证 OpenAIGenerator 可使用外部 client 并流式产出文本片段。"""
    client = FakeClient()
    generator = OpenAIGenerator(
        model="gpt-5.4-mini",
        client=client,
        request_kwargs={"max_output_tokens": 128},
    )

    chunks = list(generator.stream_generate("  prompt  "))

    assert chunks == ["你", "好", "！"]
    assert client.responses.create_calls == [
        {
            "model": "gpt-5.4-mini",
            "input": "prompt",
            "stream": True,
            "max_output_tokens": 128,
        }
    ]


def test_openai_generator_generate_collects_stream_chunks():
    """验证 generate 会消费流式输出并拼接完整答案。"""
    generator = OpenAIGenerator(client=FakeClient())

    answer = generator.generate(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ]
    )

    assert answer == "你好！"


def test_openai_generator_accepts_chat_messages_prompt():
    """验证 chat messages prompt 会被规范化后传给 OpenAI。"""
    client = FakeClient()
    generator = OpenAIGenerator(client=client)
    prompt = [
        {"role": "system", "content": " system "},
        {"role": "user", "content": " question "},
    ]

    list(generator.stream_generate(prompt))

    assert client.responses.create_calls[0]["input"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]


def test_openai_generator_rejects_invalid_init_input(monkeypatch):
    """验证 OpenAIGenerator 会拒绝非法初始化参数。"""
    monkeypatch.setattr(openai_generator, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(TypeError):
        OpenAIGenerator(model=123, client=FakeClient())

    with pytest.raises(ValueError):
        OpenAIGenerator(model="   ", client=FakeClient())

    with pytest.raises(TypeError):
        OpenAIGenerator(api_key=123, client=FakeClient())

    with pytest.raises(TypeError):
        OpenAIGenerator(base_url=123, client=FakeClient())

    with pytest.raises(TypeError):
        OpenAIGenerator(request_kwargs="bad", client=FakeClient())

    with pytest.raises(ValueError):
        OpenAIGenerator()


def test_openai_generator_rejects_invalid_prompt():
    """验证 OpenAIGenerator 会拒绝非法 prompt。"""
    generator = OpenAIGenerator(client=FakeClient())

    with pytest.raises(ValueError):
        generator.generate("   ")

    with pytest.raises(TypeError):
        generator.generate(123)

    with pytest.raises(ValueError):
        generator.generate([])

    with pytest.raises(TypeError):
        generator.generate(["bad-message"])

    with pytest.raises(TypeError):
        generator.generate([{"content": "missing role"}])

    with pytest.raises(TypeError):
        generator.generate([{"role": "user"}])

    with pytest.raises(ValueError):
        generator.generate([{"role": "tool", "content": "bad role"}])

    with pytest.raises(ValueError):
        generator.generate([{"role": "user", "content": "   "}])


def test_openai_generator_ignores_stream_events_without_text_delta():
    """验证不包含文本增量的流式事件会被忽略。"""
    generator = OpenAIGenerator(client=FakeClient())

    text_delta = generator._extract_text_delta(
        {
            "choices": [
                {
                    "delta": {
                        "content": None,
                    }
                }
            ]
        }
    )

    assert text_delta is None


def test_get_openai_client_class_lazy_loads_sdk(monkeypatch):
    """验证 _get_openai_client_class 会在需要时懒加载 OpenAI SDK。"""
    fake_module = type("FakeOpenAIModule", (), {"OpenAI": FakeOpenAI})

    monkeypatch.setattr(openai_generator, "OpenAI", None)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)

    assert openai_generator._get_openai_client_class() is FakeOpenAI
