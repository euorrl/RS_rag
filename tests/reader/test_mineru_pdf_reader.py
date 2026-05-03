import pytest

from app.reader import BaseReader
from app.reader import MinerUPdfReader
from app.schemas import Document

pytestmark = pytest.mark.reader


class MockMinerUClient(BaseReader):
    """模拟 MinerUClient。"""

    def parse_file(
        self,
        path,
        *,
        language="ch",
        model_version="vlm",
        is_ocr=True,
        enable_formula=True,
        enable_table=True,
    ):
        """模拟 MinerU 文件解析。

        Args:
            path (pathlib.Path): 待解析文件路径。
            language (str): 文档语言。
            model_version (str): 模型版本。
            is_ocr (bool): 是否启用 OCR。
            enable_formula (bool): 是否启用公式识别。
            enable_table (bool): 是否启用表格识别。

        Returns:
            str: 模拟解析得到的 Markdown 文本。
        """
        assert path.name == "test.pdf"
        assert language == "ch"
        assert model_version == "vlm"
        assert is_ocr is True
        assert enable_formula is True
        assert enable_table is True

        return "# PDF 测试文档\n\nNDVI = test"


def test_mineru_pdf_reader_reads_pdf_file(tmp_path):
    """验证 MinerUPdfReader 能正确读取 PDF 并生成 Document。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf content")

    reader = MinerUPdfReader(client=MockMinerUClient())
    doc = reader.read(file_path)

    assert isinstance(doc, Document)
    assert doc.file_name == "test.pdf"
    assert doc.file_type == ".pdf"
    assert doc.text == "# PDF 测试文档\n\nNDVI = test"
    assert doc.metadata["reader"] == "MinerUPdfReader"
    assert doc.metadata["source_format"] == "pdf"
    assert doc.metadata["text_format"] == "markdown"
    assert doc.metadata["char_count"] == len(doc.text)
    assert doc.document_id is not None


def test_mineru_pdf_reader_raises_error_when_file_not_found():
    """验证 PDF 文件不存在时抛出 FileNotFoundError。

    Returns:
        None
    """
    reader = MinerUPdfReader(client=MockMinerUClient())

    with pytest.raises(FileNotFoundError):
        reader.read("not_exist.pdf")
