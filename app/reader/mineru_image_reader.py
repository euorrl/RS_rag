from pathlib import Path

from app.reader.reader_base import BaseReader
from app.reader.mineru_client import MinerUClient
from app.schemas import Document


class MinerUImageReader(BaseReader):
    """基于 MinerU 精准解析 API 的图片 Reader。"""

    def __init__(self, client: MinerUClient | None = None) -> None:
        """初始化 MinerUImageReader。

        Args:
            client (MinerUClient | None): MinerU API 客户端。
                如果未提供，则在真正读取文件时再创建 MinerUClient。
        """
        self.client = client

    def _get_client(self) -> MinerUClient:
        """延迟创建 MinerUClient。"""
        if self.client is None:
            self.client = MinerUClient()
        return self.client

    def read(self, file_path: str | Path) -> Document:
        """读取图片文件，并返回标准 Document 对象。"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = self._get_client().parse_file(
            path,
            language="ch",
            model_version="vlm",
            is_ocr=True,
            enable_formula=False,
            enable_table=True,
        )

        return Document(
            source_path=str(path),
            file_name=path.name,
            file_type=path.suffix.lower(),
            text=text,
            metadata={
                "reader": "MinerUImageReader",
                "source_format": "image",
                "text_format": "markdown",
                "char_count": len(text),
            },
        )
