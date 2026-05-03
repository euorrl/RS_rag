from pathlib import Path

from app.reader import (
    BaseReader,
    MarkdownReader,
    MinerUImageReader,
    MinerUPdfReader,
    TextReader,
)
from app.schemas import Document


def get_reader(file_path: str | Path) -> BaseReader:
    """根据文件类型返回对应的 Reader 实例。

    Args:
        file_path (str | Path): 输入文件路径。

    Returns:
        BaseReader: 对应的 Reader 实例。

    Raises:
        ValueError: 文件类型不支持时抛出。
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".txt":
        return TextReader()

    if suffix == ".md":
        return MarkdownReader()

    if suffix == ".pdf":
        return MinerUPdfReader()

    if suffix in {".png", ".jpg", ".jpeg"}:
        return MinerUImageReader()

    raise ValueError(f"Unsupported file type: {suffix}")


def load_document(file_path: str | Path) -> Document:
    """统一读取入口，根据文件类型自动选择 Reader 并读取文件。

    Args:
        file_path (str | Path): 输入文件路径。

    Returns:
        Document: 解析后的文档对象。

    Raises:
        ValueError: 文件类型不支持时抛出。
        FileNotFoundError: 文件不存在时由具体 Reader 抛出。
    """
    reader = get_reader(file_path)
    return reader.read(file_path)
