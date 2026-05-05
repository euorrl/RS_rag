import pytest

from app.embedder.minilm_embedder import SentenceTransformerEmbedder

pytestmark = pytest.mark.embedder


class FakeEmbedding:
    """模拟 numpy.ndarray 的 tolist 行为。"""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeSentenceTransformer:
    """模拟 SentenceTransformer，避免真实加载模型。"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.last_texts = None
        self.last_normalize_embeddings = None

    def encode(self, texts, normalize_embeddings=True):
        self.last_texts = texts
        self.last_normalize_embeddings = normalize_embeddings

        return FakeEmbedding([[1.0, 2.0] for _ in texts])


def test_minilm_embedder_initializes_model(monkeypatch):
    """验证 MiniLM Embedder 能正确初始化模型。"""
    monkeypatch.setattr(
        "app.embedder.minilm_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = SentenceTransformerEmbedder(model_name="test-model")

    assert embedder.model_name == "test-model"
    assert embedder.normalize_embeddings is True
    assert embedder.model.model_name == "test-model"


def test_minilm_model_load_failure(monkeypatch):
    """验证模型加载失败时抛出 RuntimeError。"""

    def raise_error(model_name):
        raise OSError("load failed")

    monkeypatch.setattr(
        "app.embedder.minilm_embedder.SentenceTransformer",
        raise_error,
    )

    with pytest.raises(RuntimeError):
        SentenceTransformerEmbedder(model_name="bad-model")


# =========================
# embed_texts 正常流程
# =========================
def test_minilm_embed_texts_returns_embeddings(monkeypatch):
    """验证 embed_texts 能返回向量。"""
    monkeypatch.setattr(
        "app.embedder.minilm_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = SentenceTransformerEmbedder(normalize_embeddings=True)

    embeddings = embedder.embed_texts(["hello", "world"])

    assert embeddings == [[1.0, 2.0], [1.0, 2.0]]
    assert embedder.model.last_texts == ["hello", "world"]
    assert embedder.model.last_normalize_embeddings is True


def test_minilm_embed_texts_without_normalization(monkeypatch):
    """验证 normalize_embeddings=False 时参数传递正确。"""
    monkeypatch.setattr(
        "app.embedder.minilm_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = SentenceTransformerEmbedder(normalize_embeddings=False)

    embedder.embed_texts(["hello"])

    assert embedder.model.last_normalize_embeddings is False


def test_minilm_embed_texts_rejects_non_list(monkeypatch):
    """验证 texts 不是 list 时抛出 TypeError。"""
    monkeypatch.setattr(
        "app.embedder.minilm_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = SentenceTransformerEmbedder()

    with pytest.raises(TypeError):
        embedder.embed_texts("hello")


def test_minilm_embed_texts_rejects_non_string_item(monkeypatch):
    """验证 texts 中包含非字符串时抛出 TypeError。"""
    monkeypatch.setattr(
        "app.embedder.minilm_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = SentenceTransformerEmbedder()

    with pytest.raises(TypeError):
        embedder.embed_texts(["hello", 123])


def test_minilm_embed_texts_returns_empty_list(monkeypatch):
    """验证空输入返回空列表。"""
    monkeypatch.setattr(
        "app.embedder.minilm_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = SentenceTransformerEmbedder()

    assert embedder.embed_texts([]) == []


def test_minilm_embed_query_uses_default(monkeypatch):
    """验证 embed_query 使用 BaseEmbedder 默认实现。"""
    monkeypatch.setattr(
        "app.embedder.minilm_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = SentenceTransformerEmbedder()

    result = embedder.embed_query("NDVI是什么")

    assert result == [1.0, 2.0]
    assert embedder.model.last_texts == ["NDVI是什么"]
