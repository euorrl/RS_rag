from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.vector_store.vector_store_base import BaseVectorStore


class MilvusVectorStore(BaseVectorStore):
    """基于 Milvus 的向量数据库实现。"""

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        host: str = "localhost",
        port: str = "19530",
        dimension: int = 512,
    ) -> None:
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dimension = dimension

        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
        )

    def create_collection(self) -> None:
        """创建 Milvus collection。"""

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
