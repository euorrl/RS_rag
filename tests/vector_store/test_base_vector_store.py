import pytest

from app.vector_store import BaseVectorStore

pytestmark = pytest.mark.vector_store


class SuperCallingVectorStore(BaseVectorStore):
    """用于覆盖 BaseVectorStore 抽象方法默认实现的测试子类。"""

    def create_collection(self) -> None:
        return super().create_collection()

    def insert(self, embedded_chunks) -> None:
        return super().insert(embedded_chunks)

    def search(
        self,
        query_vector,
        top_k=5,
    ):
        return super().search(query_vector, top_k=top_k)

    def drop_collection(self) -> None:
        return super().drop_collection()


def test_base_vector_store_cannot_be_instantiated():
    """验证 BaseVectorStore 作为抽象类不能被直接实例化。"""
    with pytest.raises(TypeError):
        BaseVectorStore()


def test_base_vector_store_default_methods_raise_not_implemented():
    """验证 BaseVectorStore 默认方法体会抛出 NotImplementedError。"""
    vector_store = SuperCallingVectorStore()

    with pytest.raises(NotImplementedError):
        vector_store.create_collection()

    with pytest.raises(NotImplementedError):
        vector_store.insert([])

    with pytest.raises(NotImplementedError):
        vector_store.search([0.1, 0.2])

    with pytest.raises(NotImplementedError):
        vector_store.drop_collection()
