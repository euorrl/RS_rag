import pytest

from app.pipeline import RAGPipeline
from app.schemas import RetrievedChunk

pytestmark = pytest.mark.pipeline


class FakeRecaller:
    """模拟 Recaller，避免测试时连接真实向量数据库。"""

    def __init__(self):
        self.calls = []

    def recall(self, query, top_k=30, score_threshold=0.4):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
            }
        )
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                text="NDVI 是归一化植被指数。",
                score=0.8,
            )
        ]


class FakeReranker:
    """模拟 Reranker，避免测试时加载真实 reranker 模型。"""

    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_n=10, score_threshold=0.5):
        self.calls.append(
            {
                "query": query,
                "candidates": candidates,
                "top_n": top_n,
                "score_threshold": score_threshold,
            }
        )
        for index, candidate in enumerate(candidates[:top_n], start=1):
            candidate.score = 0.9
            candidate.rank = index
            candidate.rerank_method = "fake"
            candidate.score_details["rerank_score"] = 0.9
        return candidates[:top_n]


class FakePromptBuilder:
    """模拟 PromptBuilder，记录传入的查询、候选结果和历史消息。"""

    def __init__(self):
        self.calls = []

    def build(self, query, retrieved_chunks, history=None):
        self.calls.append(
            {
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "history": history,
            }
        )
        return [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": f"question: {query}",
            },
        ]


class FakeStringPromptBuilder:
    """模拟返回字符串 prompt 的 PromptBuilder。"""

    def build(self, query, retrieved_chunks, history=None):
        return f"string prompt: {query}"


class FakeGenerator:
    """模拟 Generator，避免测试时调用真实 LLM。"""

    def __init__(self):
        self.calls = []
        self.generate_calls = []

    def generate(self, prompt):
        self.generate_calls.append(prompt)
        return "改写后的检索问题"

    def stream_generate(self, prompt):
        self.calls.append(prompt)
        yield "测试"
        yield "答案"


class FakeQueryRewriter:
    """模拟 QueryRewriter，验证 Pipeline 会在召回前改写检索问题。"""

    def __init__(self):
        self.calls = []

    def rewrite(self, query, history=None):
        self.calls.append(
            {
                "query": query,
                "history": history,
            }
        )
        return f"改写：{query}"


def test_pipeline_reuses_initialized_components_across_turns():
    """验证 Pipeline 多轮问答会复用已经初始化的组件。"""
    recaller = FakeRecaller()
    reranker = FakeReranker()
    prompt_builder = FakePromptBuilder()
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        recaller=recaller,
        reranker=reranker,
        prompt_builder=prompt_builder,
        generator=generator,
    )

    first_answer = pipeline.ask(
        query="什么是 NDVI?",
        print_prompt=False,
        print_answer=False,
    )
    second_answer = pipeline.ask(
        query="它有什么用途?",
        print_prompt=False,
        print_answer=False,
    )

    assert first_answer == "测试答案"
    assert second_answer == "测试答案"
    assert pipeline.recaller is recaller
    assert pipeline.reranker is reranker
    assert pipeline.prompt_builder is prompt_builder
    assert pipeline.generator is generator
    assert len(recaller.calls) == 2
    assert len(reranker.calls) == 2
    assert len(prompt_builder.calls) == 2
    assert len(generator.calls) == 2
    assert pipeline.memory.has_history()


def test_pipeline_ask_prints_current_prompt_and_answer(capsys):
    """验证 ask() 默认会打印当前轮 prompt 和流式答案。"""
    pipeline = RAGPipeline(
        recaller=FakeRecaller(),
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
    )

    answer = pipeline.ask(query="什么是 3S?")

    captured = capsys.readouterr()
    assert answer == "测试答案"
    assert "Prompt" in captured.out
    assert "Role: system" in captured.out
    assert "question: 什么是 3S?" in captured.out
    assert "Answer" in captured.out
    assert "测试答案" in captured.out


def test_pipeline_ask_prints_rewritten_query_without_prompt(capsys):
    """验证 ask() 可以只打印改写后的检索问题而不打印 prompt。"""
    pipeline = RAGPipeline(
        recaller=FakeRecaller(),
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
        query_rewriter=FakeQueryRewriter(),
    )

    pipeline.ask(
        query="它有什么用途?",
        history=[{"role": "user", "content": "什么是 NDVI?"}],
        print_rewritten_query=True,
        print_prompt=False,
        print_answer=False,
    )

    captured = capsys.readouterr()
    assert "Rewritten Query" in captured.out
    assert "改写：它有什么用途?" in captured.out
    assert "Prompt" not in captured.out
    assert "Role: system" not in captured.out


def test_pipeline_ask_prints_string_prompt(capsys):
    """验证 ask() 可以打印字符串类型 prompt。"""
    pipeline = RAGPipeline(
        recaller=FakeRecaller(),
        reranker=FakeReranker(),
        prompt_builder=FakeStringPromptBuilder(),
        generator=FakeGenerator(),
    )

    pipeline.ask(
        query="什么是 NDVI?",
        enable_query_rewriter=False,
        print_prompt=True,
        print_answer=False,
    )

    captured = capsys.readouterr()
    assert "Prompt" in captured.out
    assert "string prompt: 什么是 NDVI?" in captured.out


def test_pipeline_retrieve_passes_recall_and_rerank_parameters():
    """验证 retrieve() 会透传召回和重排阶段的运行参数。"""
    recaller = FakeRecaller()
    reranker = FakeReranker()
    pipeline = RAGPipeline(
        recaller=recaller,
        reranker=reranker,
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
    )

    results = pipeline.retrieve(
        query="query",
        recall_top_k=20,
        recall_score_threshold=0.3,
        rerank_top_n=5,
        rerank_score_threshold=0.6,
    )

    assert len(results) == 1
    assert recaller.calls[0]["top_k"] == 20
    assert recaller.calls[0]["score_threshold"] == 0.3
    assert reranker.calls[0]["top_n"] == 5
    assert reranker.calls[0]["score_threshold"] == 0.6


def test_pipeline_ask_uses_rewritten_query_for_retrieval():
    """验证 ask() 会使用改写后的 query 执行召回和重排。"""
    recaller = FakeRecaller()
    reranker = FakeReranker()
    prompt_builder = FakePromptBuilder()
    query_rewriter = FakeQueryRewriter()
    history = [
        {
            "role": "user",
            "content": "什么是 NDVI?",
        },
        {
            "role": "assistant",
            "content": "NDVI 是归一化植被指数。",
        },
    ]
    pipeline = RAGPipeline(
        recaller=recaller,
        reranker=reranker,
        prompt_builder=prompt_builder,
        generator=FakeGenerator(),
        query_rewriter=query_rewriter,
    )

    pipeline.ask(
        query="它有什么用途?",
        history=history,
        print_prompt=False,
        print_answer=False,
    )

    assert query_rewriter.calls[0]["query"] == "它有什么用途?"
    assert query_rewriter.calls[0]["history"] is history
    assert recaller.calls[0]["query"] == "改写：它有什么用途?"
    assert reranker.calls[0]["query"] == "改写：它有什么用途?"
    assert prompt_builder.calls[0]["query"] == "它有什么用途?"
    assert pipeline.last_rewritten_query == "改写：它有什么用途?"


def test_pipeline_ask_can_disable_query_rewriter():
    """验证 ask() 可以关闭 query rewrite 并直接使用原始 query 检索。"""
    recaller = FakeRecaller()
    query_rewriter = FakeQueryRewriter()
    pipeline = RAGPipeline(
        recaller=recaller,
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
        query_rewriter=query_rewriter,
    )

    pipeline.ask(
        query="它有什么用途?",
        history=[{"role": "user", "content": "什么是 NDVI?"}],
        enable_query_rewriter=False,
        print_prompt=False,
        print_answer=False,
    )

    assert query_rewriter.calls == []
    assert recaller.calls[0]["query"] == "它有什么用途?"


def test_pipeline_rewrite_query_uses_memory_history_by_default():
    """验证 rewrite_query() 未显式传入 history 时会使用内部 ChatMemory。"""
    query_rewriter = FakeQueryRewriter()
    pipeline = RAGPipeline(
        recaller=FakeRecaller(),
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
        query_rewriter=query_rewriter,
    )
    pipeline.memory.add_turn("上一轮问题", "上一轮答案")

    rewritten_query = pipeline.rewrite_query(
        query="继续说明",
        history=None,
    )

    assert rewritten_query == "改写：继续说明"
    assert query_rewriter.calls[0]["history"] == [
        {
            "role": "user",
            "content": "上一轮问题",
        },
        {
            "role": "assistant",
            "content": "上一轮答案",
        },
    ]


def test_pipeline_build_prompt_uses_memory_history_by_default():
    """验证 build_prompt() 未显式传入 history 时会使用内部 ChatMemory。"""
    prompt_builder = FakePromptBuilder()
    pipeline = RAGPipeline(
        recaller=FakeRecaller(),
        reranker=FakeReranker(),
        prompt_builder=prompt_builder,
        generator=FakeGenerator(),
    )
    pipeline.memory.add_turn("上一轮问题", "上一轮答案")

    pipeline.build_prompt(
        query="当前问题",
        retrieved_chunks=[],
    )

    history = prompt_builder.calls[0]["history"]
    assert history == [
        {
            "role": "user",
            "content": "上一轮问题",
        },
        {
            "role": "assistant",
            "content": "上一轮答案",
        },
    ]


def test_pipeline_build_prompt_supports_explicit_history():
    """验证 build_prompt() 可以使用外部显式传入的 history。"""
    prompt_builder = FakePromptBuilder()
    pipeline = RAGPipeline(
        recaller=FakeRecaller(),
        reranker=FakeReranker(),
        prompt_builder=prompt_builder,
        generator=FakeGenerator(),
    )
    history = [
        {
            "role": "user",
            "content": "外部历史",
        }
    ]

    pipeline.build_prompt(
        query="当前问题",
        retrieved_chunks=[],
        history=history,
    )

    assert prompt_builder.calls[0]["history"] is history


def test_pipeline_rejects_empty_query():
    """验证 ask() 会拒绝空查询。"""
    pipeline = RAGPipeline(
        recaller=FakeRecaller(),
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
    )

    with pytest.raises(ValueError):
        pipeline.ask(query="   ")


def test_pipeline_rejects_non_string_query():
    """验证 ask() 会拒绝非字符串 query。"""
    pipeline = RAGPipeline(
        recaller=FakeRecaller(),
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
    )

    with pytest.raises(TypeError):
        pipeline.ask(query=123)


def test_pipeline_clear_memory():
    """验证 clear_memory() 会清空内部 ChatMemory。"""
    pipeline = RAGPipeline(
        recaller=FakeRecaller(),
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
    )
    pipeline.memory.add_turn("问题", "答案")

    pipeline.clear_memory()

    assert not pipeline.memory.has_history()


def test_pipeline_uses_existing_embedder_and_vector_store(monkeypatch):
    """验证 Pipeline 可以复用已初始化的 embedder 和 vector_store。"""
    embedder = object()
    vector_store = object()
    calls = []

    def fake_get_recaller(provider="vector", **kwargs):
        calls.append(kwargs)
        return FakeRecaller()

    monkeypatch.setattr("app.pipeline.rag_pipeline.get_recaller", fake_get_recaller)

    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
    )

    assert pipeline.embedder is embedder
    assert pipeline.vector_store is vector_store
    assert calls[0]["embedder"] is embedder
    assert calls[0]["vector_store"] is vector_store


def test_pipeline_factory_dependencies_are_created_only_once(monkeypatch):
    """验证未注入组件时，Pipeline 只在初始化阶段创建依赖对象。"""
    calls = {
        "embedder": 0,
        "vector_store": 0,
        "recaller": 0,
        "reranker": 0,
        "prompt_builder": 0,
        "generator": 0,
    }

    def fake_get_embedder(provider="bge", **kwargs):
        calls["embedder"] += 1
        return object()

    def fake_get_vector_store(provider="milvus", **kwargs):
        calls["vector_store"] += 1
        return object()

    def fake_get_recaller(provider="vector", **kwargs):
        calls["recaller"] += 1
        return FakeRecaller()

    def fake_get_reranker(provider="bge", **kwargs):
        calls["reranker"] += 1
        return FakeReranker()

    def fake_get_prompt_builder(provider="chat", **kwargs):
        calls["prompt_builder"] += 1
        return FakePromptBuilder()

    def fake_get_generator(provider="openai", **kwargs):
        calls["generator"] += 1
        return FakeGenerator()

    monkeypatch.setattr("app.pipeline.rag_pipeline.get_embedder", fake_get_embedder)
    monkeypatch.setattr(
        "app.pipeline.rag_pipeline.get_vector_store",
        fake_get_vector_store,
    )
    monkeypatch.setattr("app.pipeline.rag_pipeline.get_recaller", fake_get_recaller)
    monkeypatch.setattr("app.pipeline.rag_pipeline.get_reranker", fake_get_reranker)
    monkeypatch.setattr(
        "app.pipeline.rag_pipeline.get_prompt_builder",
        fake_get_prompt_builder,
    )
    monkeypatch.setattr("app.pipeline.rag_pipeline.get_generator", fake_get_generator)

    pipeline = RAGPipeline()
    pipeline.ask("第一轮", print_prompt=False, print_answer=False)
    pipeline.ask("第二轮", print_prompt=False, print_answer=False)

    assert calls == {
        "embedder": 1,
        "vector_store": 1,
        "recaller": 1,
        "reranker": 1,
        "prompt_builder": 1,
        "generator": 1,
    }


def test_pipeline_passes_explicit_vector_store_connection_kwargs(monkeypatch):
    """验证 Pipeline 会把显式向量库连接参数传给 VectorStore。"""
    calls = {}

    def fake_get_embedder(provider="bge", **kwargs):
        return object()

    def fake_get_vector_store(provider="milvus", **kwargs):
        calls["provider"] = provider
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("app.pipeline.rag_pipeline.get_embedder", fake_get_embedder)
    monkeypatch.setattr(
        "app.pipeline.rag_pipeline.get_vector_store",
        fake_get_vector_store,
    )
    monkeypatch.setattr(
        "app.pipeline.rag_pipeline.get_recaller",
        lambda provider="vector", **kwargs: FakeRecaller(),
    )

    RAGPipeline(
        collection_name="demo_chunks",
        host="127.0.0.1",
        port="19530",
        dimension=2,
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
    )

    assert calls == {
        "provider": "milvus",
        "kwargs": {
            "collection_name": "demo_chunks",
            "host": "127.0.0.1",
            "port": "19530",
            "dimension": 2,
        },
    }
