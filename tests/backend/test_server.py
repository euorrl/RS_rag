import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from backend import server  # noqa: E402

pytestmark = pytest.mark.backend


class FakeGenerator:
    """模拟流式生成器。"""

    def stream_generate(self, prompt):
        yield "遥感"
        yield "是"
        yield "技术"


class FakePipeline:
    """模拟 RAGPipeline，避免测试时加载真实模型。"""

    def __init__(self):
        self.generator = FakeGenerator()
        self.calls = []

    def rewrite_query(self, query, history=None):
        self.calls.append(("rewrite_query", query, history))
        return query

    def retrieve(self, query):
        self.calls.append(("retrieve", query))
        return ["chunk"]

    def build_prompt(self, query, retrieved_chunks, history=None):
        self.calls.append(("build_prompt", query, retrieved_chunks, history))
        return "prompt"


def test_health_reports_pipeline_ready(monkeypatch):
    """验证服务启动后 health 接口会报告 pipeline 已就绪。"""
    pipeline = FakePipeline()
    monkeypatch.setattr(server, "create_pipeline", lambda: pipeline)
    app = server.create_app()

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "pipeline_ready": True,
        "session_count": 0,
    }


def test_chat_stream_returns_sse_chunks(monkeypatch):
    """验证 chat stream 接口会返回 SSE 增量文本。"""
    pipeline = FakePipeline()
    monkeypatch.setattr(server, "create_pipeline", lambda: pipeline)
    monkeypatch.setattr(server, "create_session_id", lambda: "session-1")
    app = server.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            json={"question": "什么是遥感"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: session\ndata: {"session_id": "session-1"}' in response.text
    assert "event: delta\ndata: 遥感" in response.text
    assert "event: delta\ndata: 是" in response.text
    assert "event: done\ndata: {}" in response.text
    assert pipeline.calls == [
        ("rewrite_query", "什么是遥感", []),
        ("retrieve", "什么是遥感"),
        ("build_prompt", "什么是遥感", ["chunk"], []),
    ]


def test_chat_stream_reuses_session_history(monkeypatch):
    """验证相同 session_id 的请求会复用上一轮对话历史。"""
    pipeline = FakePipeline()
    monkeypatch.setattr(server, "create_pipeline", lambda: pipeline)
    app = server.create_app()

    with TestClient(app) as client:
        first_response = client.post(
            "/api/chat/stream",
            json={
                "session_id": "session-1",
                "question": "什么是遥感",
            },
        )
        second_response = client.post(
            "/api/chat/stream",
            json={
                "session_id": "session-1",
                "question": "它有什么作用",
            },
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert pipeline.calls[3] == (
        "rewrite_query",
        "它有什么作用",
        [
            {"role": "user", "content": "什么是遥感"},
            {"role": "assistant", "content": "遥感是技术"},
        ],
    )


def test_clear_session_removes_chat_memory(monkeypatch):
    """验证清除会话接口会删除对应聊天记忆。"""
    pipeline = FakePipeline()
    monkeypatch.setattr(server, "create_pipeline", lambda: pipeline)
    app = server.create_app()

    with TestClient(app) as client:
        client.post(
            "/api/chat/stream",
            json={
                "session_id": "session-1",
                "question": "什么是遥感",
            },
        )
        response = client.delete("/api/chat/session/session-1")
        health_response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "session-1",
        "cleared": True,
    }
    assert health_response.json()["session_count"] == 0


def test_chat_stream_rejects_blank_question(monkeypatch):
    """验证空白问题会被拒绝。"""
    monkeypatch.setattr(server, "create_pipeline", FakePipeline)
    app = server.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            json={"question": "   "},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "问题不能为空"


def test_format_sse_supports_multiline_data():
    """验证多行文本会被格式化为合法 SSE data 行。"""
    assert server.format_sse("第一行\n第二行", event="delta") == (
        "event: delta\n" "data: 第一行\n" "data: 第二行\n\n"
    )
