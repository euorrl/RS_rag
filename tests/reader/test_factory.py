from pathlib import Path

import pytest

from app.reader import (
    MarkdownReader,
    MinerUImageReader,
    MinerUPdfReader,
    TextReader,
)
from app.reader.factory import get_reader, load_document

pytestmark = pytest.mark.reader


def test_factory_returns_text_reader():
    """验证工厂函数能根据 .txt 后缀返回 TextReader。

    Returns:
        None
    """
    reader = get_reader("test.txt")

    assert isinstance(reader, TextReader)


def test_factory_returns_markdown_reader():
    """验证工厂函数能根据 .md 后缀返回 MarkdownReader。

    Returns:
        None
    """
    reader = get_reader("test.md")

    assert isinstance(reader, MarkdownReader)


def test_factory_returns_pdf_reader():
    """验证工厂函数能根据 .pdf 后缀返回 MinerUPdfReader。

    Returns:
        None
    """
    reader = get_reader("test.pdf")

    assert isinstance(reader, MinerUPdfReader)


def test_factory_returns_image_reader():
    """验证工厂函数能根据图片后缀返回 MinerUImageReader。

    Returns:
        None
    """
    assert isinstance(get_reader("test.png"), MinerUImageReader)
    assert isinstance(get_reader("test.jpg"), MinerUImageReader)
    assert isinstance(get_reader("test.jpeg"), MinerUImageReader)


def test_factory_supports_uppercase_suffix():
    """验证工厂函数支持大写文件后缀。

    Returns:
        None
    """
    reader = get_reader("test.MD")

    assert isinstance(reader, MarkdownReader)


def test_factory_supports_path_object():
    """验证工厂函数支持 Path 类型输入。

    Returns:
        None
    """
    reader = get_reader(Path("test.txt"))

    assert isinstance(reader, TextReader)


def test_factory_raises_error_for_unsupported_type():
    """验证不支持的文件类型会抛出 ValueError。

    Returns:
        None
    """
    with pytest.raises(ValueError):
        get_reader("test.csv")


def test_load_document_reads_txt_file(tmp_path):
    """验证 load_document 能自动选择 TextReader 并读取文件。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "test.txt"
    content = "这是一个测试文档。"
    file_path.write_text(content, encoding="utf-8")

    doc = load_document(file_path)

    assert doc.file_name == "test.txt"
    assert doc.file_type == ".txt"
    assert doc.text == content
    assert doc.metadata["reader"] == "TextReader"
    assert doc.metadata["source_format"] == "txt"
    assert doc.metadata["text_format"] == "markdown"


def test_load_document_reads_markdown_file(tmp_path):
    """验证 load_document 能自动选择 MarkdownReader 并读取文件。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "test.md"
    content = "# 标题\n\n这是一个 Markdown 测试文档。"
    file_path.write_text(content, encoding="utf-8")

    doc = load_document(file_path)

    assert doc.file_name == "test.md"
    assert doc.file_type == ".md"
    assert doc.text == content
    assert doc.metadata["reader"] == "MarkdownReader"
    assert doc.metadata["source_format"] == "markdown"
    assert doc.metadata["text_format"] == "markdown"


def test_load_document_raises_error_for_unsupported_type():
    """验证 load_document 遇到不支持的文件类型时抛出 ValueError。

    Returns:
        None
    """
    with pytest.raises(ValueError):
        load_document("test.csv")
