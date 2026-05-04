import pytest

from app.chunker import chunk_document
from app.schemas import Chunk, Document

pytestmark = pytest.mark.chunker


def make_document(text: str) -> Document:
    """创建测试用 Document。"""
    return Document(
        source_path="data/test.md",
        file_name="test.md",
        file_type=".md",
        text=text,
        metadata={
            "reader": "MarkdownReader",
            "source_format": "markdown",
            "text_format": "markdown",
            "char_count": len(text),
        },
    )


def test_chunk_document_returns_chunks():
    """验证 chunk_document 可以正常返回 Chunk 列表。"""
    document = make_document("# Title\n\nContent.")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text.startswith("# Title")


def test_chunk_document_passes_kwargs():
    """验证 chunk_document 可以将参数透传给 MarkdownChunker。"""
    text = "# Title\n\n" + "遥感数据处理。" * 100
    document = make_document(text)

    chunks = chunk_document(
        document,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_chunk_document_raises_error_for_invalid_input():
    """验证 chunk_document 会透传底层输入类型异常。"""
    with pytest.raises(TypeError, match="document 必须是 app.schemas.Document 类型"):
        chunk_document("not a document")
