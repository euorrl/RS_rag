import pytest

pymilvus = pytest.importorskip("pymilvus")

from app.schemas import EmbeddedChunk, RetrievedChunk  # noqa: E402
from app.vector_store.milvus_store import MilvusVectorStore  # noqa: E402

pytestmark = pytest.mark.vector_store


def make_store(dimension: int = 2) -> MilvusVectorStore:
    """创建不触发真实 Milvus 连接的 MilvusVectorStore 测试实例。"""
    store = MilvusVectorStore.__new__(MilvusVectorStore)
    store.collection_name = "test_chunks"
    store.host = "localhost"
    store.port = "19530"
    store.dimension = dimension
    return store


def make_embedded_chunk(
    embedding: list[float] | None = None,
) -> EmbeddedChunk:
    """创建测试用 EmbeddedChunk。"""
    return EmbeddedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="NDVI 是一种植被指数。",
        embedding=embedding or [0.1, 0.2],
        metadata={"chunk_index": 0},
    )


def test_milvus_store_initializes_connection(monkeypatch):
    """验证初始化时会建立 Milvus 连接并保存配置。"""
    calls = []

    def fake_connect(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "app.vector_store.milvus_store.connections.connect",
        fake_connect,
    )

    store = MilvusVectorStore(
        collection_name="demo",
        host="127.0.0.1",
        port="19530",
        dimension=2,
    )

    assert store.collection_name == "demo"
    assert store.host == "127.0.0.1"
    assert store.port == "19530"
    assert store.dimension == 2
    assert calls == [
        {
            "alias": "default",
            "host": "127.0.0.1",
            "port": "19530",
        }
    ]


def test_milvus_store_rejects_invalid_dimension():
    """验证 dimension 不合法时会抛出 ValueError。"""
    with pytest.raises(ValueError):
        MilvusVectorStore(dimension=0)


def test_create_collection_returns_when_collection_exists(monkeypatch):
    """验证 collection 已存在时不会重复创建。"""
    store = make_store()

    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.has_collection",
        lambda name: True,
    )

    def fail_collection(*args, **kwargs):
        raise AssertionError("Collection should not be created")

    monkeypatch.setattr(
        "app.vector_store.milvus_store.Collection",
        fail_collection,
    )

    store.create_collection()


def test_create_collection_creates_schema_and_index(monkeypatch):
    """验证 create_collection 会创建字段、schema 和向量索引。"""
    store = make_store(dimension=2)
    fields = []
    schemas = []
    collections = []

    class FakeDataType:
        INT64 = "INT64"
        VARCHAR = "VARCHAR"
        FLOAT_VECTOR = "FLOAT_VECTOR"
        JSON = "JSON"

    class FakeCollection:
        def __init__(self, name, schema=None):
            self.name = name
            self.schema = schema
            self.index_calls = []
            collections.append(self)

        def create_index(self, **kwargs):
            self.index_calls.append(kwargs)

    def fake_field_schema(**kwargs):
        fields.append(kwargs)
        return kwargs

    def fake_collection_schema(**kwargs):
        schemas.append(kwargs)
        return kwargs

    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.has_collection",
        lambda name: False,
    )
    monkeypatch.setattr("app.vector_store.milvus_store.DataType", FakeDataType)
    monkeypatch.setattr(
        "app.vector_store.milvus_store.FieldSchema",
        fake_field_schema,
    )
    monkeypatch.setattr(
        "app.vector_store.milvus_store.CollectionSchema",
        fake_collection_schema,
    )
    monkeypatch.setattr(
        "app.vector_store.milvus_store.Collection",
        FakeCollection,
    )

    store.create_collection()

    assert [field["name"] for field in fields] == [
        "id",
        "chunk_id",
        "document_id",
        "text",
        "embedding",
        "metadata",
    ]
    assert fields[4]["dim"] == 2
    assert schemas[0]["fields"] == fields
    assert collections[0].name == "test_chunks"
    assert collections[0].schema == schemas[0]
    assert collections[0].index_calls[0]["field_name"] == "embedding"
    assert collections[0].index_calls[0]["index_params"]["metric_type"] == "COSINE"


def test_get_collection_returns_collection_by_name(monkeypatch):
    """验证 _get_collection 会按当前 collection_name 获取 collection。"""
    store = make_store()
    calls = []

    def fake_collection(name):
        calls.append(name)
        return {"name": name}

    monkeypatch.setattr(
        "app.vector_store.milvus_store.Collection",
        fake_collection,
    )

    collection = store._get_collection()

    assert collection == {"name": "test_chunks"}
    assert calls == ["test_chunks"]


def test_insert_writes_columnar_data_and_flushes():
    """验证 insert 会按 schema 字段顺序写入列式数据并 flush。"""
    store = make_store(dimension=2)
    chunk = make_embedded_chunk()
    calls = []

    class FakeCollection:
        def __init__(self):
            self.data = None
            self.flushed = False

        def insert(self, data):
            self.data = data

        def flush(self):
            self.flushed = True

    collection = FakeCollection()
    store.create_collection = lambda: calls.append("create_collection")
    store._get_collection = lambda: collection

    store.insert([chunk])

    assert calls == ["create_collection"]
    assert collection.data == [
        ["chunk-1"],
        ["doc-1"],
        ["NDVI 是一种植被指数。"],
        [[0.1, 0.2]],
        [{"chunk_index": 0}],
    ]
    assert collection.flushed is True


def test_insert_returns_when_input_is_empty():
    """验证空列表不会触发写入。"""
    store = make_store()
    store.create_collection = lambda: pytest.fail("should not create collection")

    store.insert([])


def test_insert_rejects_invalid_input():
    """验证 insert 会拒绝非法输入。"""
    store = make_store()

    with pytest.raises(TypeError):
        store.insert("not-list")

    with pytest.raises(TypeError):
        store.insert(["not-embedded-chunk"])

    with pytest.raises(ValueError):
        store.insert([make_embedded_chunk(embedding=[0.1])])


def test_search_returns_empty_list_when_collection_missing(monkeypatch):
    """验证 collection 不存在时 search 返回空列表。"""
    store = make_store()
    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.has_collection",
        lambda name: False,
    )

    assert store.search([0.1, 0.2]) == []


def test_search_loads_collection_and_formats_results(monkeypatch):
    """验证 search 会加载 collection 并格式化 Milvus 命中结果。"""
    store = make_store()

    class FakeHit:
        id = 10
        distance = 0.98
        entity = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "text": "NDVI 是一种植被指数。",
            "metadata": {"chunk_index": 0},
        }

    class FakeCollection:
        def __init__(self):
            self.loaded = False
            self.search_kwargs = None

        def load(self):
            self.loaded = True

        def search(self, **kwargs):
            self.search_kwargs = kwargs
            return [[FakeHit()]]

    collection = FakeCollection()
    store._get_collection = lambda: collection

    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.has_collection",
        lambda name: True,
    )

    results = store.search([0.1, 0.2], top_k=3)

    assert collection.loaded is True
    assert collection.search_kwargs["data"] == [[0.1, 0.2]]
    assert collection.search_kwargs["anns_field"] == "embedding"
    assert collection.search_kwargs["limit"] == 3
    assert collection.search_kwargs["output_fields"] == store.output_fields
    assert results == [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="NDVI 是一种植被指数。",
            score=0.98,
            metadata={"chunk_index": 0},
            milvus_id=10,
        )
    ]


def test_search_formats_missing_metadata_as_empty_dict(monkeypatch):
    """验证 Milvus 命中结果缺少 metadata 时会返回空字典。"""
    store = make_store()

    class FakeHit:
        id = 10
        distance = 0.98
        entity = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "text": "NDVI 是一种植被指数。",
            "metadata": None,
        }

    class FakeCollection:
        def load(self):
            return None

        def search(self, **kwargs):
            return [[FakeHit()]]

    store._get_collection = lambda: FakeCollection()

    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.has_collection",
        lambda name: True,
    )

    results = store.search([0.1, 0.2])

    assert results[0].metadata == {}


def test_search_rejects_invalid_query():
    """验证 search 会拒绝非法查询参数。"""
    store = make_store()

    with pytest.raises(TypeError):
        store.search("not-vector")

    with pytest.raises(ValueError):
        store.search([0.1])

    with pytest.raises(TypeError):
        store.search([0.1, "bad"])

    with pytest.raises(ValueError):
        store.search([0.1, 0.2], top_k=0)


def test_get_chunk_text_by_id_returns_none_when_collection_missing(monkeypatch):
    """验证 collection 不存在时按 chunk_id 查询原文会返回 None。"""
    store = make_store()
    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.has_collection",
        lambda name: False,
    )

    assert store.get_chunk_text_by_id("chunk-1") is None


def test_get_chunk_text_by_id_loads_collection_and_returns_text(monkeypatch):
    """验证按 chunk_id 查询时会加载 collection 并返回对应原文。"""
    store = make_store()

    class FakeCollection:
        def __init__(self):
            self.loaded = False
            self.query_kwargs = None

        def load(self):
            self.loaded = True

        def query(self, **kwargs):
            self.query_kwargs = kwargs
            return [{"text": "NDVI 是一种植被指数。"}]

    collection = FakeCollection()
    store._get_collection = lambda: collection

    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.has_collection",
        lambda name: True,
    )

    text = store.get_chunk_text_by_id('chunk-"1"')

    assert text == "NDVI 是一种植被指数。"
    assert collection.loaded is True
    assert collection.query_kwargs == {
        "expr": 'chunk_id == "chunk-\\"1\\""',
        "output_fields": ["text"],
        "limit": 1,
    }


def test_get_chunk_text_by_id_returns_none_when_chunk_missing(monkeypatch):
    """验证按 chunk_id 查询未命中时返回 None。"""
    store = make_store()

    class FakeCollection:
        def load(self):
            return None

        def query(self, **kwargs):
            return []

    store._get_collection = lambda: FakeCollection()
    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.has_collection",
        lambda name: True,
    )

    assert store.get_chunk_text_by_id("chunk-404") is None


def test_get_chunk_text_by_id_rejects_invalid_chunk_id():
    """验证按 chunk_id 查询原文会拒绝非法参数。"""
    store = make_store()

    with pytest.raises(TypeError):
        store.get_chunk_text_by_id(123)

    with pytest.raises(ValueError):
        store.get_chunk_text_by_id("")


def test_drop_collection_only_drops_existing_collection(monkeypatch):
    """验证 drop_collection 只删除已存在的 collection。"""
    store = make_store()
    dropped = []

    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.has_collection",
        lambda name: True,
    )
    monkeypatch.setattr(
        "app.vector_store.milvus_store.utility.drop_collection",
        lambda name: dropped.append(name),
    )

    store.drop_collection()

    assert dropped == ["test_chunks"]
