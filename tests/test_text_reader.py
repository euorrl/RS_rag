from pathlib import Path

import pytest

from app.reader import TextReader
from app.reader.text_reader import _read_text_safe
from app.schemas import Document


pytestmark = pytest.mark.reader


def test_text_reader_reads_txt_file(tmp_path):
    """验证 TextReader 能正确读取 TXT 文件并生成 Document。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "test.txt"
    content = "这是一个遥感考研助手测试文档。\nNDVI = test"
    file_path.write_text(content, encoding="utf-8")

    reader = TextReader()
    doc = reader.read(file_path)

    assert isinstance(doc, Document)
    assert doc.file_name == "test.txt"
    assert doc.file_type == ".txt"
    assert doc.text == content
    assert "遥感考研助手" in doc.text
    assert doc.metadata["reader"] == "TextReader"
    assert doc.metadata["source_type"] == "text"
    assert doc.metadata["extra"]["line_count"] == 2
    assert doc.document_id is not None


def test_text_reader_metadata_contains_source_info(tmp_path):
    """验证 metadata 中的 source 与 extra 信息正确。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "test.txt"
    content = "test content"
    file_path.write_text(content, encoding="utf-8")

    reader = TextReader()
    doc = reader.read(file_path)

    source = doc.metadata["source"]
    extra = doc.metadata["extra"]

    assert source["path"] == str(file_path)
    assert source["file_name"] == "test.txt"
    assert source["file_type"] == ".txt"
    assert extra["char_count"] == len(content)
    assert extra["line_count"] == 1


def test_text_reader_reads_empty_file(tmp_path):
    """验证 TextReader 能正确处理空文件。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    reader = TextReader()
    doc = reader.read(file_path)

    assert doc.text == ""
    assert doc.metadata["extra"]["char_count"] == 0
    assert doc.metadata["extra"]["line_count"] == 1


def test_text_reader_raises_error_when_file_not_found():
    """验证文件不存在时抛出 FileNotFoundError。

    Returns:
        None
    """
    reader = TextReader()

    with pytest.raises(FileNotFoundError):
        reader.read("not_exist.txt")


def test_read_text_safe_falls_back_to_latin_1(tmp_path):
    """验证 _read_text_safe 能在必要时使用 latin-1 编码。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "latin1.txt"
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
    file_path = tmp_path / "broken.txt"
    file_path.write_text("content", encoding="utf-8")

    def raise_unicode_error(self, encoding=None):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "mock error")

    monkeypatch.setattr(Path, "read_text", raise_unicode_error)

    with pytest.raises(ValueError):
        _read_text_safe(file_path)
