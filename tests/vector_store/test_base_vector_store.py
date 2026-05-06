import pytest

from app.vector_store import BaseVectorStore

pytestmark = pytest.mark.vector_store


def test_base_vector_store_cannot_be_instantiated():
    """验证 BaseVectorStore 作为抽象类不能被直接实例化。"""
    with pytest.raises(TypeError):
        BaseVectorStore()
