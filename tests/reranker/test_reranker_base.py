import pytest

from app.reranker import BaseReranker

pytestmark = pytest.mark.reranker


def test_base_reranker_cannot_be_instantiated():
    """验证 BaseReranker 作为抽象基类不能被直接实例化。"""
    with pytest.raises(TypeError):
        BaseReranker()


def test_subclass_without_rerank_cannot_be_instantiated():
    """验证未实现 rerank 方法的子类不能被实例化。"""

    class MissingRerank(BaseReranker):
        pass

    with pytest.raises(TypeError):
        MissingRerank()


def test_subclass_with_rerank_can_be_instantiated():
    """验证实现 rerank 方法的子类可以正常实例化和调用。"""

    class SimpleReranker(BaseReranker):
        def rerank(
            self,
            query: str,
            candidates: list,
            top_n: int = 10,
            score_threshold: float | None = None,
        ):
            return candidates[:top_n]

    reranker = SimpleReranker()

    assert reranker.rerank("query", []) == []
