import pytest

from app.recaller import BaseRecaller

pytestmark = pytest.mark.recaller


def test_base_recaller_cannot_be_instantiated():
    """验证 BaseRecaller 作为抽象基类不能被直接实例化。"""
    with pytest.raises(TypeError):
        BaseRecaller()


def test_subclass_without_recall_cannot_be_instantiated():
    """验证未实现 recall 方法的子类不能被实例化。"""

    class MissingRecall(BaseRecaller):
        pass

    with pytest.raises(TypeError):
        MissingRecall()


def test_subclass_with_recall_can_be_instantiated():
    """验证实现 recall 方法的子类可以正常实例化和调用。"""

    class SimpleRecaller(BaseRecaller):
        def recall(
            self,
            query: str,
            top_k: int = 30,
            score_threshold: float | None = 0.4,
        ):
            return []

    recaller = SimpleRecaller()

    assert recaller.recall("hello") == []
