from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """用户问答请求。"""

    question: str = Field(..., min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=128)
