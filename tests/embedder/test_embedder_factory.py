import pytest

from app.embedder import BGEEmbedder, SentenceTransformerEmbedder
from app.embedder.embedder_factory import embed_chunks, get_embedder
from app.schemas import Chunk

pytestmark = pytest.mark.embedder


class FakeEmbedding:
    """模拟 numpy.ndarray 的 tolist 行为。"""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeSentenceTransformer:
    """模拟 SentenceTransformer，避免测试时真实下载模型。"""

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return FakeEmbedding([1.0, 2.0])

        return FakeEmbedding([[1.0, 2.0] for _ in texts])


class FakeBGEEmbedder:
    """模拟 BGEEmbedder。"""

    model_name = "fake-bge"
    normalize_embeddings = True
    use_instruction = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_texts(self, texts):
        return [[1.0, 2.0] for _ in texts]


class FakeMiniLMEmbedder:
    """模拟 MiniLM Embedder。"""

    model_name = "fake-minilm"
    normalize_embeddings = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_texts(self, texts):
        return [[3.0, 4.0] for _ in texts]


class BadCountEmbedder:
    """模拟返回 embedding 数量错误的 Embedder。"""

    model_name = "bad-count"

    def __init__(self, **kwargs):
        pass

    def embed_texts(self, texts):
        return []


def test_get_embedder_returns_bge_embedder(monkeypatch):
    """验证 get_embedder 默认返回 BGEEmbedder。"""
    monkeypatch.setattr(
        "app.embedder.bge_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = get_embedder()

    assert isinstance(embedder, BGEEmbedder)


def test_get_embedder_returns_minilm_embedder(monkeypatch):
    """验证 provider=minilm 时返回 SentenceTransformerEmbedder。"""
    monkeypatch.setattr(
        "app.embedder.minilm_embedder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = get_embedder(provider="minilm")

    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_get_embedder_provider_case_insensitive(monkeypatch):
    """验证 provider 大小写不敏感。"""
    monkeypatch.setattr(
        "app.embedder.embedder_factory.BGEEmbedder",
        FakeBGEEmbedder,
    )

    embedder = get_embedder(provider="BGE")

    assert isinstance(embedder, FakeBGEEmbedder)


def test_get_embedder_rejects_non_string_provider():
    """验证 provider 不是字符串时抛出 TypeError。"""
    with pytest.raises(TypeError):
        get_embedder(provider=123)


def test_get_embedder_rejects_unsupported_provider():
    """验证 provider 不支持时抛出 ValueError。"""
    with pytest.raises(ValueError):
        get_embedder(provider="unknown")


def test_embed_chunks_with_bge(monkeypatch):
    """验证 embed_chunks 能使用 BGE 将 Chunk 转换为 EmbeddedChunk。"""
    monkeypatch.setattr(
        "app.embedder.embedder_factory.BGEEmbedder",
        FakeBGEEmbedder,
    )

    chunks = [
        Chunk(
            document_id="doc-1",
            text="NDVI 是一种植被指数。",
            metadata={"chunk_index": 0},
        )
    ]

    embedded_chunks = embed_chunks(chunks, provider="bge")

    assert len(embedded_chunks) == 1
    assert embedded_chunks[0].chunk_id == chunks[0].chunk_id
    assert embedded_chunks[0].document_id == "doc-1"
    assert embedded_chunks[0].text == "NDVI 是一种植被指数。"
    assert embedded_chunks[0].embedding == [1.0, 2.0]
    assert embedded_chunks[0].metadata["chunk_index"] == 0
    assert embedded_chunks[0].metadata["embedding_provider"] == "bge"
    assert embedded_chunks[0].metadata["embedding_model"] == "fake-bge"
    assert embedded_chunks[0].metadata["normalize_embeddings"] is True
    assert embedded_chunks[0].metadata["use_instruction"] is False


def test_embed_chunks_with_minilm(monkeypatch):
    """验证 embed_chunks 能使用 MiniLM 将 Chunk 转换为 EmbeddedChunk。"""
    monkeypatch.setattr(
        "app.embedder.embedder_factory.SentenceTransformerEmbedder",
        FakeMiniLMEmbedder,
    )

    chunks = [
        Chunk(
            document_id="doc-1",
            text="hello",
            metadata={"chunk_index": 0},
        )
    ]

    embedded_chunks = embed_chunks(chunks, provider="minilm")

    assert len(embedded_chunks) == 1
    assert embedded_chunks[0].chunk_id == chunks[0].chunk_id
    assert embedded_chunks[0].document_id == "doc-1"
    assert embedded_chunks[0].text == "hello"
    assert embedded_chunks[0].embedding == [3.0, 4.0]
    assert embedded_chunks[0].metadata["chunk_index"] == 0
    assert embedded_chunks[0].metadata["embedding_provider"] == "minilm"
    assert embedded_chunks[0].metadata["embedding_model"] == "fake-minilm"
    assert embedded_chunks[0].metadata["normalize_embeddings"] is True
    assert "use_instruction" not in embedded_chunks[0].metadata


def test_embed_chunks_returns_empty_list():
    """验证空 Chunk 列表直接返回空列表。"""
    assert embed_chunks([]) == []


def test_embed_chunks_rejects_non_list():
    """验证 chunks 不是 list 时抛出 TypeError。"""
    with pytest.raises(TypeError):
        embed_chunks("not-list")


def test_embed_chunks_rejects_non_chunk_item():
    """验证 chunks 中包含非 Chunk 元素时抛出 TypeError。"""
    with pytest.raises(TypeError):
        embed_chunks(["not-chunk"])


def test_embed_chunks_rejects_non_string_provider():
    """验证 provider 不是字符串时抛出 TypeError。"""
    with pytest.raises(TypeError):
        embed_chunks([], provider=123)


def test_embed_chunks_raises_when_embedding_count_mismatch(monkeypatch):
    """验证 embedding 数量与 chunk 数量不一致时抛出 RuntimeError。"""
    monkeypatch.setattr(
        "app.embedder.embedder_factory.BGEEmbedder",
        BadCountEmbedder,
    )

    chunks = [
        Chunk(document_id="doc-1", text="text", metadata={}),
    ]

    with pytest.raises(RuntimeError):
        embed_chunks(chunks, provider="bge")
