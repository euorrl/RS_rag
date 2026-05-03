from pathlib import Path

from app.reader.base import BaseReader
from app.schemas import Document


def _read_text_safe(path: Path) -> str:
    """安全读取文本文件，自动尝试多种编码。

    Args:
        path (Path): 文件路径。

    Returns:
        str: 文件内容。

    Raises:
        ValueError: 如果无法解析编码。
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解析文件编码: {path}")


class MarkdownReader(BaseReader):
    """Markdown 文件读取器（支持 .md）。"""

    def read(self, file_path: str | Path) -> Document:
        """读取 Markdown 文件并返回 Document。

        Args:
            file_path (str | Path): 文件路径。

        Returns:
            Document: 标准化文档对象。

        Raises:
            FileNotFoundError: 文件不存在时抛出。
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        text = _read_text_safe(path)

        return Document(
            source_path=str(path),
            file_name=path.name,
            file_type=path.suffix.lower(),
            text=text,
            metadata={
                "reader": "MarkdownReader",
                "source_format": "markdown",
                "text_format": "markdown",
                "char_count": len(text),
            },
        )
