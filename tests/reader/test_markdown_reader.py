from pathlib import Path

import pytest

from app.reader import MarkdownReader
from app.reader.markdown_reader import _read_text_safe
from app.schemas import Document

pytestmark = pytest.mark.reader


def test_markdown_reader_reads_md_file(tmp_path):
    """验证 MarkdownReader 能正确读取 Markdown 文件并生成 Document。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "test.md"
    content = "# 标题\n\n这是一个测试文档。\nNDVI = test"
    file_path.write_text(content, encoding="utf-8")

    reader = MarkdownReader()
    doc = reader.read(file_path)

    assert isinstance(doc, Document)
    assert doc.file_name == "test.md"
    assert doc.file_type == ".md"
    assert doc.text == content
    assert "标题" in doc.text
    assert doc.metadata["reader"] == "MarkdownReader"
    assert doc.metadata["source_format"] == "markdown"
    assert doc.metadata["text_format"] == "markdown"
    assert doc.metadata["char_count"] == len(content)
    assert doc.document_id is not None


def test_markdown_reader_reads_empty_file(tmp_path):
    """验证 MarkdownReader 能正确处理空文件。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "empty.md"
    file_path.write_text("", encoding="utf-8")

    reader = MarkdownReader()
    doc = reader.read(file_path)

    assert doc.file_name == "empty.md"
    assert doc.file_type == ".md"
    assert doc.text == ""
    assert doc.metadata["reader"] == "MarkdownReader"
    assert doc.metadata["source_format"] == "markdown"
    assert doc.metadata["text_format"] == "markdown"
    assert doc.metadata["char_count"] == 0
    assert doc.document_id is not None


def test_markdown_reader_raises_error_when_file_not_found():
    """验证文件不存在时抛出 FileNotFoundError。

    Returns:
        None
    """
    reader = MarkdownReader()

    with pytest.raises(FileNotFoundError):
        reader.read("not_exist.md")


def test_read_text_safe_falls_back_to_latin_1(tmp_path):
    """验证 _read_text_safe 能在必要时使用 latin-1 编码。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "latin1.md"
    content = "café"
    file_path.write_bytes(content.encode("latin-1"))

    assert _read_text_safe(file_path) == content


def test_read_text_safe_raises_value_error(monkeypatch, tmp_path):
    """验证所有编码都失败时抛出 ValueError。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "broken.md"
    file_path.write_text("content", encoding="utf-8")

    def raise_unicode_error(self, encoding=None):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "mock error")

    monkeypatch.setattr(Path, "read_text", raise_unicode_error)

    with pytest.raises(ValueError):
        _read_text_safe(file_path)
