from app.pipeline import RAGPipeline


def print_section(title: str) -> None:
    """打印阶段标题，方便观察每轮 RAG 流程。"""
    print()
    print("*" * 80)
    print(title)


def main() -> None:
    pipeline = RAGPipeline()

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

        pipeline.ask(query=query)


if __name__ == "__main__":
    main()
