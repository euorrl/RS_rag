from pathlib import Path

from app.schemas import Document


class BaseReader:
    """Reader 抽象基类。

    所有具体 Reader 都应继承该类，并实现 read 方法。
    """

    def read(self, file_path: str | Path) -> Document:
        """读取文件并返回标准化的 Document 对象。

        Args:
            file_path (str | Path): 输入文件路径。

        Returns:
            Document: 解析后的文档对象。

        Raises:
            NotImplementedError: 子类未实现该方法时抛出。
        """
        raise NotImplementedError
