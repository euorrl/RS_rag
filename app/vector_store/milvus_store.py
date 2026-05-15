import json
import os
from numbers import Real
from typing import Any

from dotenv import load_dotenv
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.schemas import EmbeddedChunk, RetrievedChunk
from app.vector_store.vector_store_base import BaseVectorStore

DEFAULT_MILVUS_MODE = "local"
DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_MILVUS_DB_NAME = "default"
DEFAULT_MILVUS_COLLECTION_NAME = "rag_chunks"


def _resolve_milvus_uri(
    *,
    host: str | None = None,
    port: str | None = None,
) -> str:
    """根据旧版显式参数或环境变量解析 Milvus URI。

    Args:
        host: 可选的旧版 host 参数。
        port: 可选的旧版 port 参数。

    Returns:
        Milvus 连接 URI。
    """
    if host or port:
        host_value = host or "localhost"
        port_value = port or "19530"
        if host_value.startswith(("http://", "https://")):
            return host_value if port is None else f"{host_value}:{port_value}"
        return f"http://{host_value}:{port_value}"

    return os.getenv("MILVUS_URI", DEFAULT_MILVUS_URI)


class MilvusVectorStore(BaseVectorStore):
    """基于 Milvus 的向量数据库实现。

    该类负责把已经向量化的 ``EmbeddedChunk`` 写入 Milvus，并基于查询向量
    返回最相似的文本块。它只处理向量库读写，不负责文本切分或 embedding 生成。
    """

    output_fields = ["chunk_id", "document_id", "text", "metadata"]

    def __init__(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: str | None = None,
        uri: str | None = None,
        token: str | None = None,
        db_name: str | None = None,
        mode: str | None = None,
        dimension: int = 512,
    ) -> None:
        """初始化 MilvusVectorStore 并建立 Milvus 连接。

        Args:
            collection_name: Milvus collection 名称，默认读取
                ``MILVUS_COLLECTION_NAME``。
            host: 兼容旧调用的 Milvus 主机名。优先使用 ``uri`` 或
                ``MILVUS_URI``。
            port: 兼容旧调用的 Milvus 端口。优先使用 ``uri`` 或
                ``MILVUS_URI``。
            uri: Milvus/Zilliz Cloud 连接地址，默认读取 ``MILVUS_URI``。
            token: Zilliz Cloud 或远程 Milvus 令牌，默认读取 ``MILVUS_TOKEN``。
            db_name: Milvus 数据库名称，默认读取 ``MILVUS_DB_NAME``。
            mode: Milvus 模式标识，默认读取 ``MILVUS_MODE``。
            dimension: 向量维度，必须与 embedding 模型输出维度一致。

        Raises:
            ValueError: 当 dimension 不合法时抛出。
        """
        if dimension <= 0:
            raise ValueError("dimension 必须大于 0")

        load_dotenv()

        self.mode = mode or os.getenv("MILVUS_MODE", DEFAULT_MILVUS_MODE)
        self.uri = uri or _resolve_milvus_uri(host=host, port=port)
        self.token = token if token is not None else os.getenv("MILVUS_TOKEN", "")
        self.db_name = db_name or os.getenv("MILVUS_DB_NAME", DEFAULT_MILVUS_DB_NAME)
        self.collection_name = collection_name or os.getenv(
            "MILVUS_COLLECTION_NAME",
            DEFAULT_MILVUS_COLLECTION_NAME,
        )
        self.host = host
        self.port = port
        self.dimension = dimension

        connect_kwargs = {
            "alias": "default",
            "uri": self.uri,
            "db_name": self.db_name,
        }
        if self.token.strip():
            connect_kwargs["token"] = self.token

        connections.connect(**connect_kwargs)

    def create_collection(self) -> None:
        """创建 Milvus collection 和向量索引。"""
        if utility.has_collection(self.collection_name):
            return

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=128,
            ),
            FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                max_length=128,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=8192,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="RAG chunk vector collection",
        )

        collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 8,
                    "efConstruction": 64,
                },
            },
        )

    def insert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """插入已经向量化的文本块。

        Args:
            embedded_chunks: 待写入 Milvus 的 EmbeddedChunk 列表。

        Raises:
            TypeError: 当输入不是 list[EmbeddedChunk] 时抛出。
            ValueError: 当 embedding 维度与 collection 维度不一致时抛出。
        """
        if not isinstance(embedded_chunks, list):
            raise TypeError("embedded_chunks 必须是 list[EmbeddedChunk] 类型")

        if not all(isinstance(chunk, EmbeddedChunk) for chunk in embedded_chunks):
            raise TypeError("embedded_chunks 中的元素必须全部是 EmbeddedChunk 类型")

        if not embedded_chunks:
            return

        for chunk in embedded_chunks:
            self._validate_vector(chunk.embedding, vector_name="embedding")

        self.create_collection()
        collection = self._get_collection()

        data = [
            [chunk.chunk_id for chunk in embedded_chunks],
            [chunk.document_id or "" for chunk in embedded_chunks],
            [chunk.text for chunk in embedded_chunks],
            [chunk.embedding for chunk in embedded_chunks],
            [chunk.metadata for chunk in embedded_chunks],
        ]

        collection.insert(data)
        collection.flush()

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """根据 query 向量检索最相似的文本块。

        Args:
            query_vector: 查询向量。
            top_k: 返回的最大结果数量。

        Returns:
            list[RetrievedChunk]: 检索命中的文本块列表。

        Raises:
            TypeError: 当 query_vector 不是 list[float] 时抛出。
            ValueError: 当 query_vector 维度不匹配或 top_k 不合法时抛出。
        """
        self._validate_vector(query_vector, vector_name="query_vector")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not utility.has_collection(self.collection_name):
            return []

        collection = self._get_collection()
        collection.load()

        search_results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={
                "metric_type": "COSINE",
                "params": {"ef": 64},
            },
            limit=top_k,
            output_fields=self.output_fields,
        )

        return [self._format_hit(hit) for hit in search_results[0]]

    def get_chunk_text_by_id(self, chunk_id: str) -> str | None:
        """根据 chunk_id 查询数据库中保存的 chunk 原文。
        Args:
            chunk_id: 业务侧文本块唯一标识。
        Returns:
            str | None: 命中时返回 chunk 原文；collection 不存在或未命中时返回 None。
        Raises:
            TypeError: 当 chunk_id 不是 str 时抛出。
            ValueError: 当 chunk_id 为空字符串时抛出。
        """
        if not isinstance(chunk_id, str):
            raise TypeError("chunk_id 必须是 str 类型")

        if not chunk_id:
            raise ValueError("chunk_id 不能为空")

        if not utility.has_collection(self.collection_name):
            return None

        collection = self._get_collection()
        collection.load()

        rows = collection.query(
            expr=f"chunk_id == {json.dumps(chunk_id, ensure_ascii=False)}",
            output_fields=["text"],
            limit=1,
        )

        if not rows:
            return None

        return rows[0].get("text")

    def drop_collection(self) -> None:
        """删除当前 collection，主要用于测试或重建索引。"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

    def _get_collection(self) -> Collection:
        """获取当前 collection 对象。"""
        return Collection(self.collection_name)

    def _validate_vector(
        self,
        vector: list[float],
        *,
        vector_name: str,
    ) -> None:
        """校验向量类型和维度。

        Args:
            vector: 待校验的向量。
            vector_name: 错误提示中使用的向量名称。

        Raises:
            TypeError: 当向量不是 list 或包含非数字元素时抛出。
            ValueError: 当向量维度与 collection 维度不一致时抛出。
        """
        if not isinstance(vector, list):
            raise TypeError(f"{vector_name} 必须是 list[float] 类型")

        if len(vector) != self.dimension:
            raise ValueError(f"{vector_name} 维度必须等于 {self.dimension}")

        if not all(isinstance(value, Real) for value in vector):
            raise TypeError(f"{vector_name} 中的元素必须全部是数字")

    def _format_hit(self, hit: Any) -> RetrievedChunk:
        """将 Milvus hit 对象转换为项目内部统一的 RetrievedChunk 结果。

        Args:
            hit: Milvus search 返回的单个命中结果。

        Returns:
            RetrievedChunk: 项目内部使用的检索结果。
        """
        entity = hit.entity

        return RetrievedChunk(
            chunk_id=entity.get("chunk_id"),
            document_id=entity.get("document_id"),
            text=entity.get("text"),
            score=hit.distance,
            metadata=entity.get("metadata") or {},
            milvus_id=hit.id,
        )
