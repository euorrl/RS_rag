from pathlib import Path

from app.reader.base import BaseReader
from app.schemas import Document


class MinerUReader(BaseReader):
    """MinerU Reader。

    用于解析 PDF、图片等复杂文档。
    当前先保留结构，后续再接入 MinerU 的真实解析逻辑。
    """

    def read(self, file_path: str | Path) -> Document:
        """使用 MinerU 解析文件，并转换为 Document。"""
        path = Path(file_path)

        # TODO: 后续在这里接入 MinerU。
        # 目标是把 PDF / 图片解析为 Markdown 或纯文本。
        text = ""

        return Document(
            source_path=str(path),
            file_name=path.name,
            file_type=path.suffix.lower(),
            text=text,
            metadata={
                "reader": "MinerUReader",
                "status": "not_implemented",
            },
        )
