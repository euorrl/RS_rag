from typing import Any

from app.embedder import get_embedder
from app.generator import get_generator
from app.generator.generator_base import Prompt
from app.memory import ChatMemory
from app.prompt_builder import get_prompt_builder
from app.query_rewriter import LLMQueryRewriter
from app.recaller import get_recaller
from app.reranker import get_reranker
from app.schemas import RetrievedChunk
from app.vector_store import get_vector_store


class RAGPipeline:
    """RAG 问答流水线。

    RAGPipeline 用于串联已经实现的 query_rewriter、recaller、reranker、
    prompt_builder、generator 和 memory 模块，完成一轮完整的 RAG 问答流程：
    query -> QueryRewriter -> VectorRecaller -> BGEReranker -> PromptBuilder
    -> Generator -> answer。

    该类会在初始化时创建并持有各个模型或服务对象，后续每一轮问答都会复用已经初始化的对象，
    避免反复加载 embedding 模型、reranker 模型和 LLM client。

    RAGPipeline 只负责流程编排，不负责：
    - 读取和切分文档
    - 写入向量数据库
    - 实现新的召回、重排或生成算法
    - 直接修改各子模块的核心逻辑
    """

    def __init__(
        self,
        recaller_provider: str = "vector",
        reranker_provider: str = "bge",
        prompt_builder_provider: str = "chat",
        generator_provider: str = "openai",
        embedder_provider: str = "bge",
        embedder_model_name: str = "BAAI/bge-small-zh-v1.5",
        normalize_embeddings: bool = True,
        use_instruction: bool = False,
        vector_store_provider: str = "milvus",
        collection_name: str | None = None,
        host: str | None = None,
        port: str | None = None,
        dimension: int = 512,
        reranker_model_name: str = "BAAI/bge-reranker-base",
        prompt_system_prompt: str | None = None,
        prompt_max_context_chars: int | None = 12000,
        prompt_allow_general_fallback: bool = True,
        generator_model: str = "gpt-5.4-mini",
        generator_api_key: str | None = None,
        generator_base_url: str | None = None,
        generator_client: Any | None = None,
        generator_request_kwargs: dict[str, Any] | None = None,
        memory_max_turns: int = 5,
        enable_query_rewriter: bool = True,
        embedder_kwargs: dict[str, Any] | None = None,
        vector_store_kwargs: dict[str, Any] | None = None,
        reranker_kwargs: dict[str, Any] | None = None,
        prompt_builder_kwargs: dict[str, Any] | None = None,
        generator_kwargs: dict[str, Any] | None = None,
        embedder: Any | None = None,
        vector_store: Any | None = None,
        recaller: Any | None = None,
        reranker: Any | None = None,
        prompt_builder: Any | None = None,
        generator: Any | None = None,
        memory: ChatMemory | None = None,
        query_rewriter: Any | None = None,
    ) -> None:
        """初始化 RAGPipeline。

        Args:
            recaller_provider (str): Recaller 类型，默认 "vector"。
            reranker_provider (str): Reranker 类型，默认 "bge"。
            prompt_builder_provider (str): PromptBuilder 类型，默认 "chat"。
            generator_provider (str): Generator 类型，默认 "openai"。
            embedder_provider (str): 自动创建 Embedder 时使用的 provider，默认 "bge"。
            embedder_model_name (str): embedding 模型名称。
            normalize_embeddings (bool): embedding 是否归一化。
            use_instruction (bool): embedding 是否使用 instruction prefix。
            vector_store_provider (str): 自动创建 VectorStore 时使用的 provider，默认 "milvus"。
            collection_name (str): Milvus collection 名称。
            host (str): Milvus 服务地址。
            port (str): Milvus 服务端口。
            dimension (int): 向量维度。
            reranker_model_name (str): reranker 模型名称，默认 "BAAI/bge-reranker-base"。
            prompt_system_prompt (str | None): 自定义系统提示词。
            prompt_max_context_chars (int | None): 参考资料部分的最大字符数。
            prompt_allow_general_fallback (bool): 资料不足时是否允许通用知识补充。
            generator_model (str): LLM 模型名称。
            generator_api_key (str | None): OpenAI API Key。
            generator_base_url (str | None): OpenAI 兼容接口地址。
            generator_client (Any | None): 已初始化的 OpenAI client，主要用于测试或自定义 client。
            generator_request_kwargs (dict[str, Any] | None): 传递给模型接口的额外请求参数。
            memory_max_turns (int): ChatMemory 最多保留的最近完整问答轮数。
            enable_query_rewriter (bool): 是否启用 query rewrite，默认 True。
            embedder_kwargs (dict[str, Any] | None): 额外传递给 get_embedder() 的参数。
            vector_store_kwargs (dict[str, Any] | None): 额外传递给 get_vector_store() 的参数。
            reranker_kwargs (dict[str, Any] | None): 额外传递给 get_reranker() 的参数。
            prompt_builder_kwargs (dict[str, Any] | None): 额外传递给
                get_prompt_builder() 的参数。
            generator_kwargs (dict[str, Any] | None): 额外传递给 get_generator() 的参数。
            embedder (Any | None): 已初始化的 Embedder，传入后不会重复创建。
            vector_store (Any | None): 已初始化的 VectorStore，传入后不会重复创建。
            recaller (Any | None): 已初始化的 Recaller，传入后不会重复创建。
            reranker (Any | None): 已初始化的 Reranker，传入后不会重复创建。
            prompt_builder (Any | None): 已初始化的 PromptBuilder，传入后不会重复创建。
            generator (Any | None): 已初始化的 Generator，传入后不会重复创建。
            memory (ChatMemory | None): 已初始化的 ChatMemory，传入后不会重复创建。
            query_rewriter (Any | None): 已初始化的 QueryRewriter，传入后不会重复创建。
        """
        self.embedder = embedder
        self.vector_store = vector_store

        if recaller is None:
            self.embedder = self._get_or_create_embedder(
                embedder=self.embedder,
                provider=embedder_provider,
                model_name=embedder_model_name,
                normalize_embeddings=normalize_embeddings,
                use_instruction=use_instruction,
                embedder_kwargs=embedder_kwargs,
            )
            self.vector_store = self._get_or_create_vector_store(
                vector_store=self.vector_store,
                provider=vector_store_provider,
                collection_name=collection_name,
                host=host,
                port=port,
                dimension=dimension,
                vector_store_kwargs=vector_store_kwargs,
            )
            self.recaller = get_recaller(
                provider=recaller_provider,
                embedder=self.embedder,
                vector_store=self.vector_store,
            )
        else:
            self.recaller = recaller

        self.reranker = reranker or self._create_reranker(
            provider=reranker_provider,
            model_name=reranker_model_name,
            reranker_kwargs=reranker_kwargs,
        )
        self.prompt_builder = prompt_builder or self._create_prompt_builder(
            provider=prompt_builder_provider,
            system_prompt=prompt_system_prompt,
            max_context_chars=prompt_max_context_chars,
            allow_general_fallback=prompt_allow_general_fallback,
            prompt_builder_kwargs=prompt_builder_kwargs,
        )
        self.generator = generator or self._create_generator(
            provider=generator_provider,
            model=generator_model,
            api_key=generator_api_key,
            base_url=generator_base_url,
            client=generator_client,
            request_kwargs=generator_request_kwargs,
            generator_kwargs=generator_kwargs,
        )
        self.memory = memory or ChatMemory(max_turns=memory_max_turns)
        self.enable_query_rewriter = enable_query_rewriter
        self.query_rewriter = query_rewriter or LLMQueryRewriter(
            generator=self.generator,
        )
        self.last_rewritten_query: str | None = None

    def retrieve(
        self,
        query: str,
        recall_top_k: int = 30,
        recall_score_threshold: float | None = 0.4,
        rerank_top_n: int = 10,
        rerank_score_threshold: float | None = 0.5,
    ) -> list[RetrievedChunk]:
        """执行召回和重排，返回最终候选 chunks。

        Args:
            query (str): 用户查询文本。
            recall_top_k (int): 向量召回阶段返回的最大候选数量，默认 30。
            recall_score_threshold (float | None): 向量召回阶段最低分数阈值，默认 0.4。
            rerank_top_n (int): 重排后返回的最大结果数量，默认 10。
            rerank_score_threshold (float | None): 重排阶段最低分数阈值，默认 0.5。

        Returns:
            list[RetrievedChunk]: 重排后的候选 chunks。
        """
        recall_results = self.recaller.recall(
            query=query,
            top_k=recall_top_k,
            score_threshold=recall_score_threshold,
        )
        return self.reranker.rerank(
            query=query,
            candidates=recall_results,
            top_n=rerank_top_n,
            score_threshold=rerank_score_threshold,
        )

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> Prompt:
        """根据查询、候选 chunks 和历史对话构建当前轮 prompt。

        Args:
            query (str): 用户查询文本。
            retrieved_chunks (list[RetrievedChunk]): 已重排的候选 chunks。
            history (list[dict[str, str]] | None): 可选历史消息。
                如果不传，则自动使用内部 ChatMemory 中保存的历史。

        Returns:
            Prompt: 当前轮 prompt，可以是字符串或 chat messages。
        """
        if history is None:
            history = self.memory.get_messages()

        return self.prompt_builder.build(
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
        )

    def rewrite_query(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        enable_query_rewriter: bool | None = None,
    ) -> str:
        """根据历史对话改写检索用 query。

        Args:
            query (str): 当前用户原始问题。
            history (list[dict[str, str]] | None): 历史对话 messages。
                如果不传，则自动使用内部 ChatMemory 中保存的历史。
            enable_query_rewriter (bool | None): 是否启用 query rewrite。
                如果为 None，则使用初始化时的 enable_query_rewriter。

        Returns:
            str: 用于召回和重排的检索问题。
        """
        if history is None:
            history = self.memory.get_messages()

        if enable_query_rewriter is None:
            enable_query_rewriter = self.enable_query_rewriter

        if not enable_query_rewriter:
            return query.strip()

        return self.query_rewriter.rewrite(
            query=query,
            history=history,
        )

    def ask(
        self,
        query: str,
        recall_top_k: int = 30,
        recall_score_threshold: float | None = 0.4,
        rerank_top_n: int = 10,
        rerank_score_threshold: float | None = 0.5,
        history: list[dict[str, str]] | None = None,
        enable_query_rewriter: bool | None = None,
        save_to_memory: bool = True,
        print_rewritten_query: bool = False,
        print_prompt: bool = True,
        print_answer: bool = True,
    ) -> str:
        """执行一轮完整 RAG 问答，并返回最终答案。

        Args:
            query (str): 用户查询文本。
            recall_top_k (int): 向量召回阶段返回的最大候选数量，默认 30。
            recall_score_threshold (float | None): 向量召回阶段最低分数阈值，默认 0.4。
            rerank_top_n (int): 重排后返回的最大结果数量，默认 10。
            rerank_score_threshold (float | None): 重排阶段最低分数阈值，默认 0.5。
            history (list[dict[str, str]] | None): 可选历史消息。
                如果不传，则自动使用内部 ChatMemory 中保存的历史。
            enable_query_rewriter (bool | None): 是否启用 query rewrite。
                如果为 None，则使用初始化时的 enable_query_rewriter。
            save_to_memory (bool): 是否在答案生成完成后写入内部 ChatMemory。
            print_rewritten_query (bool): 是否打印改写后的检索问题。
            print_prompt (bool): 是否打印当前轮 prompt。
            print_answer (bool): 是否流式打印当前轮答案。

        Returns:
            str: LLM 生成的完整答案。
        """
        if not isinstance(query, str):
            raise TypeError("query 必须是 str 类型")

        query = query.strip()
        if not query:
            raise ValueError("query 不能为空字符串")

        if history is None:
            history = self.memory.get_messages()

        rewritten_query = self.rewrite_query(
            query=query,
            history=history,
            enable_query_rewriter=enable_query_rewriter,
        )
        self.last_rewritten_query = rewritten_query

        if print_rewritten_query:
            self._print_section("Rewritten Query")
            print(rewritten_query)

        retrieved_chunks = self.retrieve(
            query=rewritten_query,
            recall_top_k=recall_top_k,
            recall_score_threshold=recall_score_threshold,
            rerank_top_n=rerank_top_n,
            rerank_score_threshold=rerank_score_threshold,
        )
        prompt = self.build_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
        )

        if print_prompt:
            self._print_section("Prompt")
            self._print_prompt(prompt)

        if print_answer:
            self._print_section("Answer")

        answer_chunks = []
        for chunk in self.generator.stream_generate(prompt=prompt):
            answer_chunks.append(chunk)
            if print_answer:
                print(chunk, end="", flush=True)

        if print_answer:
            print()

        answer = "".join(answer_chunks).strip()
        if save_to_memory and answer:
            self.memory.add_turn(
                user_content=query,
                assistant_content=answer,
            )

        return answer

    def clear_memory(self) -> None:
        """清空内部 ChatMemory。"""
        self.memory.clear()

    def _get_or_create_embedder(
        self,
        embedder: Any | None,
        provider: str,
        model_name: str,
        normalize_embeddings: bool,
        use_instruction: bool,
        embedder_kwargs: dict[str, Any] | None,
    ) -> Any:
        """获取或创建 Embedder。"""
        if embedder is not None:
            return embedder

        kwargs = {
            "model_name": model_name,
            "normalize_embeddings": normalize_embeddings,
            "use_instruction": use_instruction,
        }
        kwargs.update(embedder_kwargs or {})
        return get_embedder(provider=provider, **kwargs)

    def _get_or_create_vector_store(
        self,
        vector_store: Any | None,
        provider: str,
        collection_name: str | None,
        host: str | None,
        port: str | None,
        dimension: int,
        vector_store_kwargs: dict[str, Any] | None,
    ) -> Any:
        """获取或创建 VectorStore。"""
        if vector_store is not None:
            return vector_store

        kwargs = {
            "dimension": dimension,
        }
        if collection_name is not None:
            kwargs["collection_name"] = collection_name
        if host is not None:
            kwargs["host"] = host
        if port is not None:
            kwargs["port"] = port
        kwargs.update(vector_store_kwargs or {})
        return get_vector_store(provider=provider, **kwargs)

    def _create_reranker(
        self,
        provider: str,
        model_name: str,
        reranker_kwargs: dict[str, Any] | None,
    ) -> Any:
        """创建 Reranker。"""
        kwargs = {"model_name": model_name}
        kwargs.update(reranker_kwargs or {})
        return get_reranker(provider=provider, **kwargs)

    def _create_prompt_builder(
        self,
        provider: str,
        system_prompt: str | None,
        max_context_chars: int | None,
        allow_general_fallback: bool,
        prompt_builder_kwargs: dict[str, Any] | None,
    ) -> Any:
        """创建 PromptBuilder。"""
        kwargs = {
            "system_prompt": system_prompt,
            "max_context_chars": max_context_chars,
            "allow_general_fallback": allow_general_fallback,
        }
        kwargs.update(prompt_builder_kwargs or {})
        return get_prompt_builder(provider=provider, **kwargs)

    def _create_generator(
        self,
        provider: str,
        model: str,
        api_key: str | None,
        base_url: str | None,
        client: Any | None,
        request_kwargs: dict[str, Any] | None,
        generator_kwargs: dict[str, Any] | None,
    ) -> Any:
        """创建 Generator。"""
        kwargs = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "client": client,
            "request_kwargs": request_kwargs,
        }
        kwargs.update(generator_kwargs or {})
        return get_generator(provider=provider, **kwargs)

    def _print_prompt(self, prompt: Prompt) -> None:
        """打印当前轮 prompt。"""
        if isinstance(prompt, str):
            print(prompt)
            return

        for message in prompt:
            print(f"Role: {message['role']}")
            print(message["content"])
            print("-" * 80)

    def _print_section(self, title: str) -> None:
        """打印阶段标题。"""
        print()
        print("*" * 80)
        print(title)
