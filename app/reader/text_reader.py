from pathlib import Path

from app.reader import BaseReader
from app.schemas import Document


def _read_text_safe(path: Path) -> str:
    """安全读取文本文件，自动尝试多种编码。

    Args:
        path (Path): 文件路径。

    Returns:
        str: 读取到的文本内容。

    Raises:
        ValueError: 无法解析文件编码时抛出。
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解析文件编码: {path}")


class TextReader(BaseReader):
    """纯文本 Reader。

    支持 .txt、.md 等文本文件读取，并转换为 Document。
    """

    def read(self, file_path: str | Path) -> Document:
        """读取文本文件并返回 Document。

        Args:
            file_path (str | Path): 输入文件路径。

        Returns:
            Document: 标准化后的文档对象。

        Raises:
            FileNotFoundError: 文件不存在时抛出。
            ValueError: 文件编码无法解析时抛出。
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
                "reader": "TextReader",
                "source_format": "txt",
                "text_format": "markdown",
                "char_count": len(text),
            },
        )
