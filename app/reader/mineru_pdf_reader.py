from pathlib import Path

from app.reader.base import BaseReader
from app.reader.mineru_client import MinerUClient
from app.schemas import Document


class MinerUPdfReader(BaseReader):
    """基于 MinerU 精准解析 API 的 PDF Reader。"""

    def __init__(self, client: MinerUClient | None = None) -> None:
        """初始化 MinerUPdfReader。

        Args:
            client (MinerUClient | None): MinerU API 客户端。
                如果未提供，则默认创建一个 MinerUClient。
        """
        self.client = client or MinerUClient()

    def read(self, file_path: str | Path) -> Document:
        """读取 PDF 文件，并返回标准 Document 对象。

        Args:
            file_path (str | Path): 本地 PDF 文件路径。

        Returns:
            Document: 包含 Markdown 文本的标准文档对象。

        Raises:
            FileNotFoundError: 当文件不存在时抛出。
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = self.client.parse_file(
            path,
            language="ch",
            model_version="vlm",
            is_ocr=True,
            enable_formula=True,
            enable_table=True,
        )

        return Document(
            source_path=str(path),
            file_name=path.name,
            file_type=path.suffix.lower(),
            text=text,
            metadata={
                "reader": "MinerUPdfReader",
                "source_format": "pdf",
                "text_format": "markdown",
                "char_count": len(text),
            },
        )
