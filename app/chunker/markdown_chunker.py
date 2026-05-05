from app.chunker.chunker_base import BaseChunker
from app.schemas import Chunk, Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class MarkdownChunker(BaseChunker):
    """Markdown 文档切分器。

    该组件用于将 Markdown 文档切分为适用于向量检索的 Chunk 列表。

    切分过程分为两步：
    1. 基于 Markdown 标题结构进行语义切分（如 # / ## / ###）。
    2. 对过长文本进行递归切分，保证每个 chunk 大小可控。

    输出为项目自定义的 Chunk 对象，包含文本内容及结构化 metadata，
    便于后续 embedding、检索和溯源。
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ) -> None:
        """初始化 MarkdownChunker。

        Args:
            chunk_size: 每个 chunk 的最大字符数。
            chunk_overlap: 相邻 chunk 之间的重叠字符数。
            headers_to_split_on: Markdown 标题切分规则。
                例如 [("#", "h1"), ("##", "h2")]。
                若不提供，则默认使用 h1~h3。

        Raises:
            ValueError: 当 chunk_size 或 chunk_overlap 不合法时抛出。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 必须大于等于 0")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
            or [
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """将 Document 切分为 Chunk 列表。

        Args:
            document: 输入的 Document 对象，包含 Markdown 文本。

        Returns:
            切分后的 Chunk 列表。

        Raises:
            TypeError: 当输入不是 Document，或 document.text 不是字符串时抛出。
            RuntimeError: 当底层 LangChain 切分失败时抛出。
        """
        if not isinstance(document, Document):
            raise TypeError("document 必须是 app.schemas.Document 类型")

        if not isinstance(document.text, str):
            raise TypeError("document.text 必须是 str 类型")

        if not document.text.strip():
            return []

        try:
            sections = self.header_splitter.split_text(document.text)
            langchain_chunks = self.text_splitter.split_documents(sections)
        except Exception as exc:
            raise RuntimeError(f"Markdown 文档切分失败: {exc}") from exc

        chunks: list[Chunk] = []

        for index, item in enumerate(langchain_chunks):
            headers = {
                "h1": item.metadata.get("h1"),
                "h2": item.metadata.get("h2"),
                "h3": item.metadata.get("h3"),
            }

            header_path = " > ".join(value for value in headers.values() if value)

            metadata = {
                "source_path": document.source_path,
                "file_name": document.file_name,
                "file_type": document.file_type,
                "reader": document.metadata.get("reader"),
                "source_format": document.metadata.get("source_format"),
                "text_format": document.metadata.get("text_format"),
                "document_char_count": document.metadata.get("char_count"),
                "chunk_index": index,
                "chunk_size": len(item.page_content),
                "headers": headers,
                "header_path": header_path,
            }

            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    text=item.page_content,
                    metadata=metadata,
                )
            )

        return chunks
