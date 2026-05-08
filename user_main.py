from app.prompt_builder import build_messages_prompt
from app.recaller import recall
from app.reranker import rerank


def main() -> None:
    query = "什么是3S，有哪些应用?"

    recall_results = recall(
        query=query,
        provider="vector",
        top_k=30,
        score_threshold=0.4,
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
    reranker_results = rerank(
        query=query,
        candidates=recall_results,
        provider="bge",
        top_n=10,
        score_threshold=0.5,
        model_name="BAAI/bge-reranker-v2-m3",
    )
    prompt = build_messages_prompt(
        query=query,
        retrieved_chunks=reranker_results,
        provider="chat",
        history=None,
        max_context_chars=12000,
        allow_general_fallback=True,
    )

    print(f"Query: {query}")
    print(f"Recall results: {len(recall_results)}")
    print(f"Rerank results: {len(reranker_results)}")

    print("*" * 80)
    for idx, result in enumerate(reranker_results, start=1):
        print(f"Rank: {idx}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Document ID: {result.document_id}")
        print(f"Text: {result.text}")
        print(f"Score: {result.score}")
        print(f"Score details: {result.score_details}")
        print("-" * 80)

    print("*" * 80)
    print("Messages prompt:")
    print(prompt)


if __name__ == "__main__":
    main()
