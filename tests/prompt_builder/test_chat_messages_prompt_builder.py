import pytest

from app.prompt_builder.chat_messages_prompt_builder import ChatMessagesPromptBuilder
from app.schemas import RetrievedChunk

pytestmark = pytest.mark.prompt_builder


def make_retrieved_chunks() -> list[RetrievedChunk]:
    """创建测试用 RetrievedChunk 列表。"""
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="NDVI 是归一化植被指数。",
            score=0.95,
            metadata={
                "file_name": "remote_sensing.md",
                "header_path": "遥感指数 > 植被指数",
            },
            score_details={
                "rerank_score": 0.95,
            },
        )
    ]


def test_chat_messages_prompt_builder_builds_messages():
    """验证 ChatMessagesPromptBuilder 会生成 system 和当前 user 消息。"""
    prompt_builder = ChatMessagesPromptBuilder()

    messages = prompt_builder.build("什么是 NDVI?", make_retrieved_chunks())

    assert messages[0]["role"] == "system"
    assert "你是一个严谨的知识问答助手" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "参考资料：" in messages[1]["content"]
    assert "用户问题：" in messages[1]["content"]
    assert "什么是 NDVI?" in messages[1]["content"]
    assert "来源: remote_sensing.md" in messages[1]["content"]
    assert "标题: 遥感指数 > 植被指数" in messages[1]["content"]
    assert "chunk-1" not in messages[1]["content"]
    assert "score_details" not in messages[1]["content"]


def test_chat_messages_prompt_builder_accepts_string_builder_options():
    """验证 ChatMessagesPromptBuilder 支持字符串构建器的初始化参数。"""
    prompt_builder = ChatMessagesPromptBuilder(
        system_prompt="自定义系统提示词",
        max_context_chars=20,
        allow_general_fallback=False,
    )

    messages = prompt_builder.build("什么是 NDVI?", make_retrieved_chunks())

    assert prompt_builder.allow_general_fallback is False
    assert prompt_builder.max_context_chars == 20
    assert messages[0] == {
        "role": "system",
        "content": "自定义系统提示词",
    }
    assert "...[参考资料已截断]" in messages[-1]["content"]


def test_chat_messages_prompt_builder_inserts_history_between_messages():
    """验证 history 会被插入 system message 和当前 user message 之间。"""
    prompt_builder = ChatMessagesPromptBuilder()
    history = [
        {"role": "system", "content": "old system"},
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
    ]

    messages = prompt_builder.build(
        query="什么是 NDVI?",
        retrieved_chunks=make_retrieved_chunks(),
        history=history,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] != "old system"
    assert messages[1] == {"role": "user", "content": "上一轮问题"}
    assert messages[2] == {"role": "assistant", "content": "上一轮回答"}
    assert "什么是 NDVI?" in messages[3]["content"]


def test_chat_messages_prompt_builder_rejects_invalid_history():
    """验证 ChatMessagesPromptBuilder 会拒绝非法 history。"""
    prompt_builder = ChatMessagesPromptBuilder()

    with pytest.raises(TypeError):
        prompt_builder.build("query", [], history="bad-history")

    with pytest.raises(TypeError):
        prompt_builder.build("query", [], history=["bad-message"])

    with pytest.raises(TypeError):
        prompt_builder.build("query", [], history=[{"content": "missing role"}])

    with pytest.raises(TypeError):
        prompt_builder.build("query", [], history=[{"role": "user"}])

    with pytest.raises(ValueError):
        prompt_builder.build(
            "query",
            [],
            history=[{"role": "tool", "content": "bad role"}],
        )
