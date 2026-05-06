import pytest
from langchain_core.documents import Document as LangChainDocument

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


def test_markdown_chunker_raises_value_error_for_invalid_tail_ratio():
    """验证短尾 chunk 比例不合法时抛出 ValueError。"""
    with pytest.raises(ValueError, match="min_tail_chunk_ratio 必须大于 0 且小于 1"):
        MarkdownChunker(min_tail_chunk_ratio=0)


def test_markdown_chunker_tail_ratio_must_be_larger_than_overlap_ratio():
    """验证短尾 chunk 比例必须大于 overlap 与 chunk_size 的比例。"""
    with pytest.raises(
        ValueError,
        match="min_tail_chunk_ratio 必须大于 chunk_overlap / chunk_size",
    ):
        MarkdownChunker(
            chunk_size=100,
            chunk_overlap=20,
            min_tail_chunk_ratio=0.2,
        )


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
    assert chunks[0].text == "Content."
    assert chunks[0].metadata["header_path"] == "Title"


def test_markdown_chunker_keeps_section_when_not_oversized(monkeypatch):
    """验证标题 section 未超过 chunk_size 时不会进入二次切分。"""
    document = make_document("# Title\n\nShort content.")
    chunker = MarkdownChunker(chunk_size=100, chunk_overlap=10)

    def fail_if_called(_sections):
        raise AssertionError("text splitter should not be called")

    monkeypatch.setattr(
        chunker.text_splitter,
        "split_documents",
        fail_if_called,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].text == "Short content."
    assert chunks[0].metadata["header_path"] == "Title"


def test_markdown_chunker_splits_section_only_when_oversized():
    """验证标题 section 超过 chunk_size 时才继续按长度切分。"""
    text = "# Title\n\n" + "遥感数据处理" * 30
    document = make_document(text)

    chunks = chunk_document(
        document,
        chunk_size=80,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["header_path"] == "Title" for chunk in chunks)
    assert all(len(chunk.text) <= 80 for chunk in chunks)


def test_markdown_chunker_forces_overlap_inside_oversized_section():
    """验证只有进入二次切分的 section 内部才强制 overlap。"""
    text = "# Title\n\n" + "0123456789" * 12
    document = make_document(text)

    chunks = chunk_document(
        document,
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 50 for chunk in chunks)

    for previous, current in zip(chunks, chunks[1:]):
        assert current.text.startswith(previous.text[-10:])


def test_markdown_chunker_does_not_overlap_between_header_sections():
    """验证 Markdown 标题 section 之间不会强制 overlap。"""
    text = """# Title

第一部分内容。

## Section

第二部分内容。
"""
    document = make_document(text)

    chunks = chunk_document(
        document,
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) == 2
    assert chunks[0].metadata["header_path"] == "Title"
    assert chunks[1].metadata["header_path"] == "Title > Section"
    assert not chunks[1].text.startswith(chunks[0].text[-10:])


def test_markdown_chunker_merges_short_tail_chunk_into_previous_chunk():
    """验证最后一个新增信息过少的 chunk 会合并到前一个 chunk。"""
    text = "# Title\n\n" + ("A" * 40) + ("B" * 40) + ("C" * 5)
    document = make_document(text)

    chunks = chunk_document(
        document,
        chunk_size=50,
        chunk_overlap=10,
        min_tail_chunk_ratio=0.25,
    )

    assert len(chunks) == 2
    assert chunks[-1].text.endswith("C" * 5)
    assert len(chunks[-1].text) > 50


def test_markdown_chunker_skips_empty_header_sections():
    """验证空标题 section 会被跳过。"""
    chunker = MarkdownChunker()
    sections = [
        LangChainDocument(page_content="   ", metadata={"h1": "Empty"}),
        LangChainDocument(page_content="Content.", metadata={"h1": "Title"}),
    ]

    chunks = chunker._split_oversized_sections(sections)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Content."
    assert chunks[0].metadata["h1"] == "Title"


def test_markdown_chunker_does_not_force_overlap_when_overlap_is_zero():
    """验证 chunk_overlap 为 0 时不会补充强制 overlap。"""
    chunker = MarkdownChunker(
        chunk_size=50,
        chunk_overlap=0,
        min_tail_chunk_ratio=0.25,
    )
    section_chunks = [
        LangChainDocument(page_content="A" * 50, metadata={"h1": "Title"}),
        LangChainDocument(page_content="B" * 30, metadata={"h1": "Title"}),
    ]

    chunks = chunker._add_forced_overlap(section_chunks)

    assert chunks == section_chunks
    assert chunks[1].page_content.startswith("B")


def test_markdown_chunker_keeps_single_tail_chunk_unchanged():
    """验证只有一个二次切分 chunk 时不会执行短尾合并。"""
    chunker = MarkdownChunker()
    section_chunks = [
        LangChainDocument(page_content="Content.", metadata={"h1": "Title"}),
    ]

    chunks = chunker._merge_short_tail_chunk(section_chunks)

    assert chunks == section_chunks


def test_markdown_chunker_keeps_sentence_punctuation_with_previous_chunk():
    """验证中文句号不会被切到下一个 chunk 开头或单独成块。"""
    text = (
        "# Title\n\n"
        "在遥感应用领域进行了广泛探索和应用试验研究，"
        "如云南腾冲遥感综合试验研究、长春净月潭试验研究、"
        "山西太原盆地农业遥感试验研究、东海渔业遥感试验研究等。"
        "这些试验研究都紧密地结合遥感技术的发展和应用，"
        "为大规模、多领域的应用打下基础并起到了示范作用。"
    )
    document = make_document(text)

    chunks = chunk_document(
        document,
        chunk_size=80,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(chunk.text != "。" for chunk in chunks)
    assert all(not chunk.text.startswith("。") for chunk in chunks)


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
