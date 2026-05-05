import pytest

from app.embedder import BaseEmbedder

pytestmark = pytest.mark.embedder


def test_base_embedder_cannot_be_instantiated():
    """验证 BaseEmbedder 作为抽象类不能被直接实例化。"""
    with pytest.raises(TypeError):
        BaseEmbedder()


def test_subclass_without_embed_texts_cannot_be_instantiated():
    """验证未实现 embed_texts 方法的子类不能被实例化。"""

    class BadEmbedder(BaseEmbedder):
        pass

    with pytest.raises(TypeError):
        BadEmbedder()


def test_subclass_with_embed_texts_can_be_instantiated():
    """验证实现了 embed_texts 方法的子类可以正常实例化。"""

    class GoodEmbedder(BaseEmbedder):
        def embed_texts(self, texts):
            return [[] for _ in texts]

    embedder = GoodEmbedder()
    assert isinstance(embedder, BaseEmbedder)


def test_embed_query_default_implementation():
    """验证 embed_query 默认实现调用 embed_texts。"""

    class GoodEmbedder(BaseEmbedder):
        def embed_texts(self, texts):
            return [[1.0] for _ in texts]

    embedder = GoodEmbedder()

    result = embedder.embed_query("test")

    assert result == [1.0]
