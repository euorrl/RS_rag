from pathlib import Path

from app.schemas import Document

from app.reader import (
    BaseReader,
    MarkdownReader,
    MinerUImageReader,
    MinerUPdfReader,
    TextReader,
)


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
    """读取文件并解析为统一的 Document 对象。

    该函数是 Reader 模块的统一入口，会根据文件类型自动选择
    对应的 Reader（如 TextReader、MarkdownReader、MinerUPdfReader 等），
    将不同格式的文件解析为标准化的 Document 数据结构。

    处理流程：
    1. 根据文件后缀选择合适的 Reader；
    2. 调用 Reader.read() 解析文件；
    3. 返回统一格式的 Document 对象。


    Args:
        file_path: 输入文件路径（支持 str 或 Path）。

    Returns:
        统一格式的 Document 对象。
        包含：
        - text: 解析后的文本内容（Markdown 格式）
        - source_path: 原始文件路径
        - file_name: 文件名
        - file_type: 文件类型（如 .pdf / .md / .txt）
        - metadata: Reader 生成的附加信息，例如：
            reader 类型（TextReader / MarkdownReader 等）、
            原始数据格式（source_format / text_format）、
            字符数（char_count）
        - document_id: 文档唯一标识（自动生成）

    Raises:
        ValueError: 文件类型不支持时抛出。
        FileNotFoundError: 文件不存在时由具体 Reader 抛出。
    """
    reader = get_reader(file_path)
    return reader.read(file_path)
