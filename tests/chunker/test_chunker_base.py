import pytest

from app.chunker import BaseChunker

pytestmark = pytest.mark.chunker


def test_base_chunker_cannot_be_instantiated():
    """验证 BaseChunker 作为抽象类不能被直接实例化。"""
    with pytest.raises(TypeError):
        BaseChunker()
