import pytest

from app.memory import ChatMemory
from app.schemas import ChatHistory

pytestmark = pytest.mark.memory


def fake_generate_summary(prompt, **kwargs):
    """模拟 summary 生成，避免测试时调用真实 LLM。"""
    content = prompt[-1]["content"]
    return f"summary::{content}"


def test_chat_memory_add_turn_and_get_messages():
    """验证 ChatMemory 能保存最近对话并转换为 history messages。"""
    memory = ChatMemory(max_turns=2)

    memory.add_turn(
        user_content="什么是3S？",
        assistant_content="3S 是 RS、GIS、GPS 的合称。",
    )

    assert memory.has_history()
    assert memory.get_messages() == [
        {"role": "user", "content": "什么是3S？"},
        {"role": "assistant", "content": "3S 是 RS、GIS、GPS 的合称。"},
    ]


def test_chat_memory_keeps_recent_max_turns_and_summarizes_overflow(monkeypatch):
    """验证超过 max_turns 的早期对话会被压缩到 summary 中。"""
    monkeypatch.setattr(
        "app.memory.chat_memory.generate",
        fake_generate_summary,
    )
    memory = ChatMemory(max_turns=2)

    memory.add_turn("第一问", "第一答")
    memory.add_turn("第二问", "第二答")
    memory.add_turn("第三问", "第三答")

    history = memory.get_history()

    assert len(history.recent_turns) == 2
    assert [turn.user_content for turn in history.recent_turns] == [
        "第二问",
        "第三问",
    ]
    assert history.summary is not None
    assert "第一问" in history.summary
    assert "第一答" in history.summary

    messages = memory.get_messages()
    assert messages[0]["role"] == "assistant"
    assert "以下是较早对话的摘要" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "第二问"}


def test_chat_memory_merges_existing_summary_when_overflow_again(monkeypatch):
    """验证再次溢出时会将已有 summary 与最早一轮对话重新合并。"""
    calls = []

    def fake_generate(prompt, **kwargs):
        calls.append(
            {
                "prompt": prompt,
                "kwargs": kwargs,
            }
        )
        return f"summary-{len(calls)}"

    monkeypatch.setattr(
        "app.memory.chat_memory.generate",
        fake_generate,
    )
    memory = ChatMemory(max_turns=1)

    memory.add_turn("第一问", "第一答")
    memory.add_turn("第二问", "第二答")
    memory.add_turn("第三问", "第三答")

    assert memory.get_history().summary == "summary-2"
    assert len(calls) == 2
    assert "已有摘要：" in calls[1]["prompt"][-1]["content"]
    assert "summary-1" in calls[1]["prompt"][-1]["content"]
    assert "第二问" in calls[1]["prompt"][-1]["content"]
    assert calls[1]["kwargs"] == {
        "provider": "openai",
        "model": "gpt-5.4-mini",
    }


def test_chat_memory_clear():
    """验证 ChatMemory 能清空 summary 和 recent_turns。"""
    memory = ChatMemory(max_turns=2)
    memory.add_turn("问题", "回答")

    assert memory.has_history()

    memory.clear()

    assert not memory.has_history()
    assert memory.get_messages() == []
    assert memory.get_history().summary is None
    assert memory.get_history().recent_turns == []


def test_chat_memory_get_history_returns_internal_chat_history():
    """验证 get_history 返回内部 ChatHistory 对象。"""
    memory = ChatMemory(max_turns=2)

    history = memory.get_history()

    assert isinstance(history, ChatHistory)
    assert history is memory.history


def test_chat_memory_rejects_invalid_max_turns():
    """验证 ChatMemory 会拒绝非法 max_turns。"""
    with pytest.raises(TypeError):
        ChatMemory(max_turns="5")

    with pytest.raises(ValueError):
        ChatMemory(max_turns=0)


def test_chat_memory_add_turn_reuses_conversation_turn_validation():
    """验证 add_turn 复用 ConversationTurn 的内容校验逻辑。"""
    memory = ChatMemory(max_turns=2)

    with pytest.raises(TypeError):
        memory.add_turn(123, "回答")

    with pytest.raises(TypeError):
        memory.add_turn("问题", 123)

    with pytest.raises(ValueError):
        memory.add_turn("", "回答")

    with pytest.raises(ValueError):
        memory.add_turn("问题", "")


def test_chat_memory_rejects_invalid_generated_summary(monkeypatch):
    """验证 summary 生成结果非法时会抛出异常。"""
    memory = ChatMemory(max_turns=1)
    memory.add_turn("第一问", "第一答")

    monkeypatch.setattr("app.memory.chat_memory.generate", lambda *args, **kwargs: 123)
    with pytest.raises(TypeError):
        memory.add_turn("第二问", "第二答")

    memory = ChatMemory(max_turns=1)
    memory.add_turn("第一问", "第一答")

    monkeypatch.setattr("app.memory.chat_memory.generate", lambda *args, **kwargs: " ")
    with pytest.raises(ValueError):
        memory.add_turn("第二问", "第二答")
