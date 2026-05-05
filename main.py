from app.reader import load_document
from app.chunker import chunk_document
from app.embedder import embed_chunks


def main() -> None:
    """验证 Document -> Chunk -> EmbeddedChunk 的基础流程。"""

    document = load_document("data/test_markdown.md")
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


if __name__ == "__main__":
    main()
