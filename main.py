from app.reader import load_document
from app.chunker import chunk_document
from app.embedder import embed_chunks
from app.vector_store import get_vector_store

COLLECTION_NAME = "rag_chunks"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"


def main() -> None:
    """验证 Document -> Chunk -> EmbeddedChunk -> Milvus 的基础流程。"""

    document = load_document("data/遥感导论-第1章-17-29.pdf")
    chunks = chunk_document(document)

    print("========== Chunks ==========")
    print(f"chunk 数量: {len(chunks)}")

    for chunk in chunks:
        print("-" * 40)
        print("chunk_id:", chunk.chunk_id)
        print("document_id:", chunk.document_id)
        print("text:", chunk.text[:100])
        print("metadata:", chunk.metadata)

    embedded_chunks = embed_chunks(
        chunks,
        provider="bge",
        model_name="BAAI/bge-small-zh-v1.5",
        normalize_embeddings=True,
        use_instruction=False,
    )

    print("\n========== Embedded Chunks ==========")
    print(f"embedded chunk 数量: {len(embedded_chunks)}")

    for embedded_chunk in embedded_chunks:
        print("-" * 40)
        print("chunk_id:", embedded_chunk.chunk_id)
        print("document_id:", embedded_chunk.document_id)
        print("text:", embedded_chunk.text[:100])
        print("embedding 维度:", len(embedded_chunk.embedding))
        print("embedding 前 5 个值:", embedded_chunk.embedding[:5])
        print("metadata:", embedded_chunk.metadata)

    if not embedded_chunks:
        print("\n没有可写入 Milvus 的 embedded chunks。")
        return

    vector_store = get_vector_store(
        provider="milvus",
        collection_name=COLLECTION_NAME,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        dimension=len(embedded_chunks[0].embedding),
    )
    vector_store.drop_collection()
    vector_store.insert(embedded_chunks)

    print("\n========== Milvus ==========")
    print("collection:", vector_store.collection_name)
    print(f"已写入 embedded chunk 数量: {len(embedded_chunks)}")


if __name__ == "__main__":
    main()
