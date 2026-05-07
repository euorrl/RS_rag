from app.schemas import Document, Chunk
from app.chunker.chunker_base import BaseChunker
from app.chunker.markdown_chunker import MarkdownChunker


def get_chunker(document: Document, **kwargs) -> BaseChunker:
    """根据 document 选择 chunker（未来扩展点）"""
    if not isinstance(document, Document):
        raise TypeError("document 必须是 app.schemas.Document 类型")

    text_format = document.metadata.get("text_format")

    if text_format == "markdown":
        return MarkdownChunker(**kwargs)

    raise ValueError(f"Unsupported text format: {text_format}")


def chunk_document(document: Document, **kwargs) -> list[Chunk]:
    """对 Document 进行切分，返回 Chunk 列表（支持参数透传）。

    该函数是对 MarkdownChunker 的轻量封装，用于简化调用流程。
    调用时可通过 **kwargs 向 MarkdownChunker 传递初始化参数，
    如 chunk_size、chunk_overlap、headers_to_split_on 等。

    适用于：
    - 快速调用（无需显式实例化）
    - 需要简单配置但不想直接操作类的场景

    Args:
        document: 输入的 Document 对象，包含待切分的 Markdown 文本。
        **kwargs: 传递给 MarkdownChunker 的参数，例如：
            - chunk_size (int): 每个 chunk 的最大字符数
            - chunk_overlap (int): 相邻 chunk 的重叠字符数
            - headers_to_split_on (list[tuple[str, str]]): 标题切分规则

    Returns:
        切分后的 Chunk 列表，每个 Chunk 包含：
        - text: 文本内容
        - metadata: 上下文信息（来源、标题层级、位置等）
        - document_id: 所属文档 ID

    Raises:
        ValueError: 当传入的 chunk 参数（如 chunk_size、chunk_overlap）不合法时抛出。
        TypeError: 当 document 类型不正确或 document.text 不是字符串时抛出。
        RuntimeError: 当底层切分过程（LangChain）发生异常时抛出。
    """
    chunker = get_chunker(document, **kwargs)
    return chunker.chunk(document)
