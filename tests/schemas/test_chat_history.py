import pytest

from app.schemas import ChatHistory, ConversationTurn

pytestmark = pytest.mark.schemas


# =========================
# ConversationTurn 正常流程
# =========================
def test_conversation_turn_to_messages():
    """验证 ConversationTurn 能正确转换为 Chat API history messages。"""
    turn = ConversationTurn(
        user_content="什么是3S？",
        assistant_content="3S 是 RS、GIS、GPS 的合称。",
    )

    assert turn.to_messages() == [
        {"role": "user", "content": "什么是3S？"},
        {"role": "assistant", "content": "3S 是 RS、GIS、GPS 的合称。"},
    ]


def test_conversation_turn_strips_content():
    """验证 ConversationTurn 会自动清理首尾空白字符。"""
    turn = ConversationTurn(
        user_content="  什么是3S？  ",
        assistant_content="  3S 是 RS、GIS、GPS。  ",
    )

    assert turn.user_content == "什么是3S？"
    assert turn.assistant_content == "3S 是 RS、GIS、GPS。"


# =========================
# ConversationTurn 异常输入
# =========================
def test_conversation_turn_rejects_empty_user_content():
    """验证 user_content 为空字符串时抛出 ValueError。"""
    with pytest.raises(ValueError):
        ConversationTurn(user_content="", assistant_content="回答")


def test_conversation_turn_rejects_empty_assistant_content():
    """验证 assistant_content 为空字符串时抛出 ValueError。"""
    with pytest.raises(ValueError):
        ConversationTurn(user_content="问题", assistant_content="")


def test_conversation_turn_rejects_non_string_user_content():
    """验证 user_content 不是字符串时抛出 TypeError。"""
    with pytest.raises(TypeError):
        ConversationTurn(user_content=123, assistant_content="回答")


def test_conversation_turn_rejects_non_string_assistant_content():
    """验证 assistant_content 不是字符串时抛出 TypeError。"""
    with pytest.raises(TypeError):
        ConversationTurn(user_content="问题", assistant_content=123)


# =========================
# ChatHistory 正常流程
# =========================
def test_chat_history_add_turn_and_to_messages():
    """验证 ChatHistory 能添加一轮对话并转换为 messages。"""
    history = ChatHistory()
    history.add_turn(
        user_content="什么是3S？",
        assistant_content="3S 是 RS、GIS、GPS 的合称。",
    )

    assert history.to_messages() == [
        {"role": "user", "content": "什么是3S？"},
        {"role": "assistant", "content": "3S 是 RS、GIS、GPS 的合称。"},
    ]


def test_chat_history_with_summary():
    """验证 ChatHistory 存在 summary 时会将其放在最近对话之前。"""
    history = ChatHistory(summary="之前讨论过3S的定义。")
    history.add_turn(
        user_content="它有哪些应用？",
        assistant_content="可用于车辆导航、精细农业、防灾减灾等。",
    )

    messages = history.to_messages()

    assert messages[0]["role"] == "assistant"
    assert "之前讨论过3S的定义" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "它有哪些应用？"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "可用于车辆导航、精细农业、防灾减灾等。"


def test_chat_history_clear():
    """验证 ChatHistory 能正确清空 summary 和 recent_turns。"""
    history = ChatHistory(summary="摘要")
    history.add_turn("问题", "回答")

    assert history.has_history()

    history.clear()

    assert not history.has_history()
    assert history.summary is None
    assert history.recent_turns == []


def test_chat_history_has_history_without_content():
    """验证空 ChatHistory 的 has_history 返回 False。"""
    history = ChatHistory()

    assert not history.has_history()


def test_chat_history_has_history_with_summary():
    """验证只有 summary 时 has_history 返回 True。"""
    history = ChatHistory(summary="之前讨论过3S。")

    assert history.has_history()


def test_chat_history_has_history_with_recent_turns():
    """验证存在 recent_turns 时 has_history 返回 True。"""
    history = ChatHistory()
    history.add_turn("问题", "回答")

    assert history.has_history()


# =========================
# ChatHistory 异常输入
# =========================
def test_chat_history_rejects_non_string_summary():
    """验证 summary 不是字符串或 None 时抛出 TypeError。"""
    with pytest.raises(TypeError):
        ChatHistory(summary=123)


def test_chat_history_normalizes_empty_summary_to_none():
    """验证空白 summary 会被规范化为 None。"""
    history = ChatHistory(summary="   ")

    assert history.summary is None


def test_chat_history_rejects_non_list_recent_turns():
    """验证 recent_turns 不是 list 时抛出 TypeError。"""
    with pytest.raises(TypeError):
        ChatHistory(recent_turns="not-a-list")


def test_chat_history_rejects_non_conversation_turn_item():
    """验证 recent_turns 中包含非 ConversationTurn 元素时抛出 TypeError。"""
    with pytest.raises(TypeError):
        ChatHistory(recent_turns=["bad-turn"])
