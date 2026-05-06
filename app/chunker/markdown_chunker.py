from app.chunker.chunker_base import BaseChunker
from app.schemas import Chunk, Document
from langchain_core.documents import Document as LangChainDocument
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
        min_tail_chunk_ratio: float = 0.25,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ) -> None:
        """初始化 MarkdownChunker。

        Args:
            chunk_size: 每个 chunk 的最大字符数。
            chunk_overlap: 相邻 chunk 之间的重叠字符数。
            min_tail_chunk_ratio: 最后一个二次切分 chunk 的最小长度比例。
                如果最后一个 chunk 的新增内容小于 chunk_size * 该比例，
                则合并到前一个 chunk，避免产生重复占比过高的短尾 chunk。
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

        if not 0 < min_tail_chunk_ratio < 1:
            raise ValueError("min_tail_chunk_ratio 必须大于 0 且小于 1")

        if min_tail_chunk_ratio <= chunk_overlap / chunk_size:
            raise ValueError("min_tail_chunk_ratio 必须大于 chunk_overlap / chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_tail_chunk_ratio = min_tail_chunk_ratio

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
            or [
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=True,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size - chunk_overlap if chunk_overlap else chunk_size,
            chunk_overlap=0,
            keep_separator="end",
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", ".", ",", " ", ""],
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
            langchain_chunks = self._split_oversized_sections(sections)
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

    def _split_oversized_sections(
        self,
        sections: list[LangChainDocument],
    ) -> list[LangChainDocument]:
        """只对超过 chunk_size 的标题 section 做二次切分。"""
        chunks: list[LangChainDocument] = []

        for section in sections:
            page_content = section.page_content.strip()
            if not page_content:
                continue

            normalized_section = LangChainDocument(
                page_content=page_content,
                metadata=section.metadata,
            )

            if len(page_content) <= self.chunk_size:
                chunks.append(normalized_section)
                continue

            section_chunks = self.text_splitter.split_documents([normalized_section])
            chunks.extend(self._add_forced_overlap(section_chunks))

        return chunks

    def _add_forced_overlap(
        self,
        section_chunks: list[LangChainDocument],
    ) -> list[LangChainDocument]:
        """为同一个超长 section 内部的相邻 chunk 强制补充 overlap。"""
        section_chunks = self._merge_short_tail_chunk(section_chunks)

        if self.chunk_overlap == 0 or len(section_chunks) <= 1:
            return section_chunks

        overlapped_chunks = [section_chunks[0]]

        for chunk in section_chunks[1:]:
            previous_text = overlapped_chunks[-1].page_content
            overlap_text = previous_text[-self.chunk_overlap :]
            page_content = chunk.page_content

            if overlap_text and not page_content.startswith(overlap_text):
                page_content = overlap_text + page_content

            overlapped_chunks.append(
                LangChainDocument(
                    page_content=page_content,
                    metadata=chunk.metadata,
                )
            )

        return overlapped_chunks

    def _merge_short_tail_chunk(
        self,
        section_chunks: list[LangChainDocument],
    ) -> list[LangChainDocument]:
        """将新增信息过少的最后一个 chunk 合并到前一个 chunk。"""
        if len(section_chunks) <= 1:
            return section_chunks

        tail = section_chunks[-1]
        min_tail_size = self.chunk_size * self.min_tail_chunk_ratio

        if len(tail.page_content) >= min_tail_size:
            return section_chunks

        previous = section_chunks[-2]
        merged_previous = LangChainDocument(
            page_content=previous.page_content + tail.page_content,
            metadata=previous.metadata,
        )

        return [*section_chunks[:-2], merged_previous]
