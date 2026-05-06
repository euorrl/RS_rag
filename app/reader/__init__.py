from typing import TYPE_CHECKING

from app.reader.reader_base import BaseReader

if TYPE_CHECKING:  # pragma: no cover
    from app.reader.markdown_reader import (  # noqa: F401
        MarkdownReader as MarkdownReader,
    )
    from app.reader.mineru_image_reader import (  # noqa: F401
        MinerUImageReader as MinerUImageReader,
    )
    from app.reader.mineru_pdf_reader import (  # noqa: F401
        MinerUPdfReader as MinerUPdfReader,
    )
    from app.reader.reader_factory import get_reader as get_reader  # noqa: F401
    from app.reader.reader_factory import load_document as load_document  # noqa: F401
    from app.reader.text_reader import TextReader as TextReader  # noqa: F401

__all__ = [
    "BaseReader",
    "TextReader",
    "MarkdownReader",
    "MinerUPdfReader",
    "MinerUImageReader",
    "get_reader",
    "load_document",
]


def __getattr__(name: str):
    if name == "TextReader":
        from app.reader.text_reader import TextReader

        return TextReader

    if name == "MarkdownReader":
        from app.reader.markdown_reader import MarkdownReader

        return MarkdownReader

    if name == "MinerUPdfReader":
        from app.reader.mineru_pdf_reader import MinerUPdfReader

        return MinerUPdfReader

    if name == "MinerUImageReader":
        from app.reader.mineru_image_reader import MinerUImageReader

        return MinerUImageReader

    if name in {"get_reader", "load_document"}:
        from app.reader.reader_factory import get_reader, load_document

        return {
            "get_reader": get_reader,
            "load_document": load_document,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
