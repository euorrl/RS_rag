import pytest

from app.embedder.bge_embedder import BGEEmbedder

pytestmark = pytest.mark.embedder


class FakeEmbedding:
    """模拟 numpy.ndarray 的 tolist 行为。"""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeSentenceTransformer:
    """模拟 SentenceTransformer。"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.last_input = None
        self.last_normalize_embeddings = None

    def encode(self, texts, normalize_embeddings=True):
        self.last_input = texts
        self.last_normalize_embeddings = normalize_embeddings

        if isinstance(texts, str):
            return FakeEmbedding([1.0, 2.0])

        return FakeEmbedding([[1.0, 2.0] for _ in texts])


def test_bge_embedder_initializes_model(monkeypatch):
    """验证 BGEEmbedder 能正确初始化模型。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder(model_name="test-bge")

    assert embedder.model_name == "test-bge"
    assert embedder.normalize_embeddings is True
    assert embedder.use_instruction is False
    assert embedder.model.model_name == "test-bge"


def test_bge_embed_texts_without_instruction(monkeypatch):
    """验证未启用 instruction 时，文本原样送入模型。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder(use_instruction=False)
    embeddings = embedder.embed_texts(["文本1", "文本2"])

    assert embeddings == [[1.0, 2.0], [1.0, 2.0]]
    assert embedder.model.last_input == ["文本1", "文本2"]
    assert embedder.model.last_normalize_embeddings is True


def test_bge_embed_texts_with_instruction(monkeypatch):
    """验证启用 instruction 时，文档文本添加 passage 前缀。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder(use_instruction=True)
    embeddings = embedder.embed_texts(["NDVI 是植被指数"])

    assert embeddings == [[1.0, 2.0]]
    assert embedder.model.last_input == ["passage: NDVI 是植被指数"]
    assert embedder.model.last_normalize_embeddings is True


def test_bge_embed_texts_without_normalization(monkeypatch):
    """验证 normalize_embeddings=False 时参数能正确传递。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder(normalize_embeddings=False)
    embedder.embed_texts(["文本"])

    assert embedder.model.last_normalize_embeddings is False


def test_bge_embed_query_without_instruction(monkeypatch):
    """验证未启用 instruction 时，查询文本原样送入模型。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder(use_instruction=False)
    embedding = embedder.embed_query("什么是 NDVI？")

    assert embedding == [1.0, 2.0]
    assert embedder.model.last_input == "什么是 NDVI？"
    assert embedder.model.last_normalize_embeddings is True


def test_bge_embed_query_with_instruction(monkeypatch):
    """验证启用 instruction 时，查询文本添加 query 前缀。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder(use_instruction=True)
    embedding = embedder.embed_query("什么是 NDVI？")

    assert embedding == [1.0, 2.0]
    assert embedder.model.last_input == "query: 什么是 NDVI？"
    assert embedder.model.last_normalize_embeddings is True


def test_bge_embed_texts_rejects_non_list(monkeypatch):
    """验证 texts 不是 list 时抛出 TypeError。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder()

    with pytest.raises(TypeError):
        embedder.embed_texts("hello")


def test_bge_embed_texts_rejects_non_string_item(monkeypatch):
    """验证 texts 中包含非字符串元素时抛出 TypeError。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder()

    with pytest.raises(TypeError):
        embedder.embed_texts(["hello", 123])


def test_bge_embed_texts_returns_empty_list(monkeypatch):
    """验证空文本列表直接返回空列表。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder()

    assert embedder.embed_texts([]) == []


def test_bge_embed_query_rejects_non_string(monkeypatch):
    """验证 query 不是字符串时抛出 TypeError。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder()

    with pytest.raises(TypeError):
        embedder.embed_query(123)


def test_bge_embed_query_rejects_empty_string(monkeypatch):
    """验证 query 为空字符串时抛出 ValueError。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder()

    with pytest.raises(ValueError):
        embedder.embed_query("   ")


def test_bge_model_load_failure(monkeypatch):
    """验证 BGE 模型加载失败时抛出 RuntimeError。"""

    def raise_error(model_name):
        raise OSError("load failed")

    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        raise_error,
    )

    with pytest.raises(RuntimeError):
        BGEEmbedder(model_name="bad-model")
