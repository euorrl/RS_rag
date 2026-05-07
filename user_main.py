from app.recaller import recall


def main() -> None:
    query = "植物遥感之所以可以被运用，其背后的原理是什么，是因为植物的光谱曲线具有区分性的特征吗?"
    score_threshold = 0.4

    results = recall(
        query=query,
        provider="vector",
        top_k=30,
        score_threshold=score_threshold,
        embedder_provider="bge",
        embedder_kwargs={
            "model_name": "BAAI/bge-small-zh-v1.5",
            "normalize_embeddings": True,
            "use_instruction": False,
        },
        vector_store_provider="milvus",
        vector_store_kwargs={
            "collection_name": "rag_chunks",
            "host": "localhost",
            "port": "19530",
            "dimension": 512,
        },
    )

    print(f"Query: {query}")
    print(f"Score threshold: {score_threshold}")
    print(f"Results: {len(results)}")

    if not results:
        print("No high-score chunks found. Send the original query to LLM directly.")
        return

    for result in results:
        print("=" * 80)
        print(f"rank: {result.rank}")
        print(f"score: {result.score}")
        print(f"chunk_id: {result.chunk_id}")
        print(f"document_id: {result.document_id}")
        print(f"metadata: {result.metadata}")
        print(result.text)


if __name__ == "__main__":
    main()
