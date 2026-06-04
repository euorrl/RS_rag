import json
import os
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.memory import ChatMemory
from app.pipeline import RAGPipeline
from backend.schemas import ChatRequest

PIPELINE: RAGPipeline | None = None
SESSIONS: dict[str, ChatMemory] = {}


def create_pipeline() -> RAGPipeline:
    """创建可被后端复用的 RAGPipeline 实例。"""
    return RAGPipeline(enable_query_rewriter=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """在服务启动时初始化 RAGPipeline，并在关闭时释放引用。"""
    global PIPELINE

    load_dotenv()
    SESSIONS.clear()
    PIPELINE = create_pipeline()
    yield
    PIPELINE = None
    SESSIONS.clear()


def get_pipeline() -> RAGPipeline:
    """获取已初始化的 RAGPipeline。"""
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="RAG 服务尚未初始化")
    return PIPELINE


def get_allowed_origins() -> list[str]:
    """从环境变量读取允许访问后端的前端地址。"""
    origins = os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


def create_session_id() -> str:
    """创建新的聊天会话 ID。"""
    return uuid.uuid4().hex


def get_memory(session_id: str) -> ChatMemory:
    """获取指定会话的聊天记忆，不存在时自动创建。"""
    if session_id not in SESSIONS:
        max_turns = int(os.getenv("CHAT_MEMORY_MAX_TURNS", "5"))
        SESSIONS[session_id] = ChatMemory(max_turns=max_turns)

    return SESSIONS[session_id]


def format_sse(
    data: str,
    *,
    event: str | None = None,
) -> str:
    """将文本格式化为 SSE 消息。"""
    lines = []
    if event:
        lines.append(f"event: {event}")

    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")

    return "\n".join(lines) + "\n\n"


def stream_rag_answer(
    pipeline: RAGPipeline,
    question: str,
    session_id: str,
) -> Iterator[str]:
    """执行 RAG 问答，并按会话历史以 SSE 形式流式返回答案。"""
    memory = get_memory(session_id)
    history = memory.get_messages()
    answer_chunks: list[str] = []

    try:
        yield format_sse(
            json.dumps({"session_id": session_id}, ensure_ascii=False),
            event="session",
        )
        rewritten_query = pipeline.rewrite_query(
            query=question,
            history=history,
        )
        retrieved_chunks = pipeline.retrieve(query=rewritten_query)
        prompt = pipeline.build_prompt(
            query=question,
            retrieved_chunks=retrieved_chunks,
            history=history,
        )

        for chunk in pipeline.generator.stream_generate(prompt=prompt):
            answer_chunks.append(chunk)
            yield format_sse(chunk, event="delta")

        answer = "".join(answer_chunks).strip()
        if answer:
            memory.add_turn(
                user_content=question,
                assistant_content=answer,
            )

        yield format_sse("{}", event="done")
    except Exception as exc:
        yield format_sse(str(exc), event="error")


def create_app() -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(
        title="RS RAG API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["POST", "GET", "DELETE"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """返回服务健康状态。"""
        return {
            "status": "ok",
            "pipeline_ready": PIPELINE is not None,
            "session_count": len(SESSIONS),
        }

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        """流式回答用户问题。"""
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        pipeline = get_pipeline()
        session_id = request.session_id or create_session_id()
        return StreamingResponse(
            stream_rag_answer(pipeline, question, session_id),
            media_type="text/event-stream",
        )

    @app.delete("/api/chat/session/{session_id}")
    def clear_session(session_id: str) -> dict[str, Any]:
        """清除指定聊天会话。"""
        existed = SESSIONS.pop(session_id, None) is not None
        return {
            "session_id": session_id,
            "cleared": existed,
        }

    return app


app = create_app()
