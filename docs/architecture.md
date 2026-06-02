# 系统架构

RS RAG 的核心分为两条链路：资料入库链路和在线问答链路。

## 资料入库链路

```text
原始文件
  -> Reader
  -> Document
  -> MarkdownChunker
  -> Chunk
  -> Embedder
  -> EmbeddedChunk
  -> MilvusVectorStore
```

### 数据结构

- `Document`：Reader 解析后的完整文档，包含来源、正文、metadata 和 `document_id`
- `Chunk`：切分后的文本块，包含 `chunk_id`、`document_id`、正文和切分 metadata
- `EmbeddedChunk`：带 embedding 的文本块，用于写入向量库
- `RetrievedChunk`：检索和重排阶段使用的统一结果结构，包含文本、分数、rank、来源和 score details

### 入库策略

`scripts/create_cloud_milvus_database.py` 默认处理 `data/remote_sensing_fundamentals` 下的 PDF。它会读取 PDF、调用 MinerU 解析、切分 Markdown、使用 BGE 生成 512 维向量，并写入 Milvus collection。

## 在线问答链路

```text
用户问题
  -> LLMQueryRewriter
  -> VectorRecaller
  -> BGEReranker
  -> ChatMessagesPromptBuilder
  -> OpenAIGenerator
  -> ChatMemory
```

`RAGPipeline` 位于 `app/pipeline/rag_pipeline.py`，负责串联各模块，但不负责文档解析、入库或具体算法实现。

默认流程：

1. 根据历史对话把用户问题改写为独立检索问题。
2. 使用 embedder 把检索问题转成 query vector。
3. 在 Milvus 中执行向量召回。
4. 使用 BGE reranker 对候选 chunks 重新打分和排序。
5. 把用户问题、历史对话和参考资料构造成 chat messages。
6. 调用 OpenAI Responses API 流式生成答案。
7. 生成完成后写入会话记忆。

## 默认组件

| 模块 | 默认实现 | 说明 |
| --- | --- | --- |
| Reader | `TextReader` / `MarkdownReader` / `MinerUPdfReader` / `MinerUImageReader` | 按文件后缀选择 |
| Chunker | `MarkdownChunker` | 标题切分 + 递归切分 |
| Embedder | `BGEEmbedder` | 默认 `BAAI/bge-small-zh-v1.5` |
| VectorStore | `MilvusVectorStore` | HNSW + COSINE |
| Recaller | `VectorRecaller` | 默认 `top_k=30` |
| Reranker | `BGEReranker` | 默认 BGE reranker |
| PromptBuilder | `ChatMessagesPromptBuilder` | 返回 Chat API messages |
| Generator | `OpenAIGenerator` | OpenAI Responses API |
| Memory | `ChatMemory` | 默认保留最近 5 轮 |
| QueryRewriter | `LLMQueryRewriter` | 多轮检索问题改写 |

## Web 架构

后端位于 `backend/server.py`，启动时初始化全局 `RAGPipeline`，按 `session_id` 管理 `ChatMemory`，并通过 `/api/chat/stream` 返回 Server-Sent Events。

前端位于 `frontend/src/App.vue`，负责调用流式接口、解析 SSE、展示消息、管理会话和停止生成。
