import pytest

from app.chunker import MarkdownChunker, chunk_document
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


def test_markdown_chunker_splits_by_headers():
    """验证 MarkdownChunker 能按 Markdown 标题结构切分。"""
    text = """# Title

Intro text.

## Section A

Content A.

### Section A.1

Content A.1.
"""
    document = make_document(text)

    chunker = MarkdownChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    assert all(isinstance(chunk, Chunk) for chunk in chunks)

    assert chunks[0].metadata["headers"]["h1"] == "Title"
    assert chunks[0].metadata["header_path"] == "Title"

    assert chunks[1].metadata["headers"]["h1"] == "Title"
    assert chunks[1].metadata["headers"]["h2"] == "Section A"
    assert chunks[1].metadata["header_path"] == "Title > Section A"

    assert chunks[2].metadata["headers"]["h1"] == "Title"
    assert chunks[2].metadata["headers"]["h2"] == "Section A"
    assert chunks[2].metadata["headers"]["h3"] == "Section A.1"
    assert chunks[2].metadata["header_path"] == "Title > Section A > Section A.1"


def test_markdown_chunker_preserves_source_metadata():
    """验证 Chunk metadata 中保留必要的文档来源信息。"""
    document = make_document("# Title\n\nContent.")

    chunker = MarkdownChunker()
    chunks = chunker.chunk(document)

    assert len(chunks) == 1

    metadata = chunks[0].metadata

    assert metadata["source_path"] == "data/test.md"
    assert metadata["file_name"] == "test.md"
    assert metadata["file_type"] == ".md"
    assert metadata["reader"] == "MarkdownReader"
    assert metadata["source_format"] == "markdown"
    assert metadata["text_format"] == "markdown"
    assert metadata["document_char_count"] == len(document.text)


def test_markdown_chunker_adds_chunk_metadata():
    """验证 Chunk metadata 中包含 chunk 位置信息和大小信息。"""
    document = make_document("# Title\n\nContent.")

    chunker = MarkdownChunker()
    chunks = chunker.chunk(document)

    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_size"] == len(chunks[0].text)


def test_markdown_chunker_links_chunk_to_document_id():
    """验证每个 Chunk 都能通过 document_id 追溯到原始 Document。"""
    document = make_document("# Title\n\nContent.")

    chunker = MarkdownChunker()
    chunks = chunker.chunk(document)

    assert chunks[0].document_id == document.document_id
    assert chunks[0].chunk_id is not None
    assert isinstance(chunks[0].chunk_id, str)


def test_markdown_chunker_returns_empty_list_for_empty_text():
    """验证空文本会返回空列表。"""
    document = make_document("   \n\n   ")

    chunker = MarkdownChunker()
    chunks = chunker.chunk(document)

    assert chunks == []


def test_markdown_chunker_raises_type_error_for_non_document_input():
    """验证输入不是 Document 时抛出 TypeError。"""
    chunker = MarkdownChunker()

    with pytest.raises(TypeError, match="document 必须是 app.schemas.Document 类型"):
        chunker.chunk("not a document")


def test_markdown_chunker_raises_type_error_for_non_string_text():
    """验证 document.text 不是字符串时抛出 TypeError。"""
    document = make_document("# Title\n\nContent.")
    document.text = None

    chunker = MarkdownChunker()

    with pytest.raises(TypeError, match="document.text 必须是 str 类型"):
        chunker.chunk(document)


def test_markdown_chunker_raises_value_error_for_invalid_chunk_size():
    """验证 chunk_size 不合法时抛出 ValueError。"""
    with pytest.raises(ValueError, match="chunk_size 必须大于 0"):
        MarkdownChunker(chunk_size=0)


def test_markdown_chunker_raises_value_error_for_negative_overlap():
    """验证 chunk_overlap 为负数时抛出 ValueError。"""
    with pytest.raises(ValueError, match="chunk_overlap 必须大于等于 0"):
        MarkdownChunker(chunk_overlap=-1)


def test_markdown_chunker_raises_value_error_when_overlap_not_smaller_than_size():
    """验证 chunk_overlap 大于等于 chunk_size 时抛出 ValueError。"""
    with pytest.raises(ValueError, match="chunk_overlap 必须小于 chunk_size"):
        MarkdownChunker(chunk_size=100, chunk_overlap=100)


def test_markdown_chunker_supports_custom_headers():
    """验证可以自定义 Markdown 标题切分规则。"""
    text = """# Title

Intro.

## Section

Content.
"""
    document = make_document(text)

    chunker = MarkdownChunker(
        headers_to_split_on=[
            ("#", "chapter"),
            ("##", "section"),
        ]
    )
    chunks = chunker.chunk(document)

    assert chunks[0].metadata["headers"]["h1"] is None
    assert chunks[0].metadata["headers"]["h2"] is None
    assert chunks[0].metadata["headers"]["h3"] is None

    assert chunks[0].metadata["header_path"] == ""


def test_chunk_document_factory_uses_default_chunker():
    """验证 chunk_document 便捷接口可以正常切分 Document。"""
    document = make_document("# Title\n\nContent.")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text.startswith("# Title")


def test_chunk_document_factory_accepts_kwargs():
    """验证 chunk_document 可以透传 MarkdownChunker 参数。"""
    text = "# Title\n\n" + "遥感数据处理。" * 100
    document = make_document(text)

    chunks = chunk_document(
        document,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_markdown_chunker_wraps_langchain_errors(monkeypatch):
    """验证底层 LangChain 切分失败时会包装为 RuntimeError。"""
    document = make_document("# Title\n\nContent.")
    chunker = MarkdownChunker()

    def mock_split_text(_text):
        raise ValueError("mock split error")

    monkeypatch.setattr(
        chunker.header_splitter,
        "split_text",
        mock_split_text,
    )

    with pytest.raises(RuntimeError, match="Markdown 文档切分失败"):
        chunker.chunk(document)
