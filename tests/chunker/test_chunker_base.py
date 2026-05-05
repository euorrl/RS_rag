import pytest

from app.chunker import BaseChunker
from app.schemas import Document, Chunk

pytestmark = pytest.mark.chunker


def test_base_chunker_cannot_be_instantiated():
    """验证 BaseChunker 作为抽象类不能被直接实例化。"""
    with pytest.raises(TypeError):
        BaseChunker()


def test_subclass_without_chunk_cannot_be_instantiated():
    """验证未实现 chunk 方法的子类不能被实例化。"""

    class BadChunker(BaseChunker):
        pass

    with pytest.raises(TypeError):
        BadChunker()


def test_subclass_with_chunk_can_be_instantiated():
    """验证实现了 chunk 方法的子类可以正常实例化。"""

    class GoodChunker(BaseChunker):
        def chunk(self, document: Document) -> list[Chunk]:
            return []

    chunker = GoodChunker()

    assert isinstance(chunker, BaseChunker)
