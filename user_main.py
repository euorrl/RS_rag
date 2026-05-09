from app.generator import stream_generate
from app.memory import ChatMemory
from app.prompt_builder import build_messages_prompt
from app.recaller import recall
from app.reranker import rerank


def print_section(title: str) -> None:
    """打印阶段标题，方便观察每轮 RAG 流程。"""
    print()
    print("*" * 80)
    print(title)


def print_prompt(prompt: list[dict[str, str]]) -> None:
    """打印 chat messages prompt。"""
    for message in prompt:
        print(f"Role: {message['role']}")
        print(message["content"])
        print("-" * 80)


def run_rag_once(query: str, memory: ChatMemory) -> str:
    """执行单轮 RAG 问答，并返回助手最终答案。"""
    print_section("Recall")
    print("Running vector recall...", flush=True)
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
    print(f"Recall results: {len(recall_results)}")

    print_section("Rerank")
    print("Running BGE reranker...", flush=True)
    reranker_results = rerank(
        query=query,
        candidates=recall_results,
        provider="bge",
        top_n=10,
        score_threshold=0.5,
        model_name="BAAI/bge-reranker-base",
    )
    print(f"Rerank results: {len(reranker_results)}")

    prompt = build_messages_prompt(
        query=query,
        retrieved_chunks=reranker_results,
        provider="chat",
        history=memory.get_messages(),
        max_context_chars=12000,
        allow_general_fallback=True,
    )

    print_section("Prompt")
    print_prompt(prompt)

    print_section("Answer")
    answer_chunks = []
    for chunk in stream_generate(
        prompt=prompt,
        provider="openai",
        model="gpt-5.4-mini",
    ):
        answer_chunks.append(chunk)
        print(chunk, end="", flush=True)
    print()

    return "".join(answer_chunks).strip()


def main() -> None:
    memory = ChatMemory(max_turns=5)

    print("RAG 多轮问答已启动。输入 exit 或 quit 退出。")
    while True:
        query = input("\n请输入问题：").strip()

        if query.lower() in {"exit", "quit"}:
            print("已退出。")
            break

        if not query:
            print("问题不能为空。")
            continue

        print_section("Query")
        print(query)

        answer = run_rag_once(query=query, memory=memory)
        if answer:
            memory.add_turn(
                user_content=query,
                assistant_content=answer,
            )


if __name__ == "__main__":
    main()
