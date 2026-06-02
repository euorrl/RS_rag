# 模块说明

本页按 RAG 链路顺序说明 `app/` 下的核心模块。每个模块都尽量保持单一职责：上游只把标准数据结构交给下游，具体算法和外部服务封装在各自模块中。

## 数据结构

位置：`app/schemas/`

核心数据结构贯穿整个项目：

| 类型 | 用途 | 主要字段 |
| --- | --- | --- |
| `Document` | Reader 解析后的完整文档 | `source_path`、`file_name`、`file_type`、`text`、`metadata`、`document_id` |
| `Chunk` | 切分后的文本块 | `chunk_id`、`document_id`、`text`、`metadata` |
| `EmbeddedChunk` | 已向量化的文本块 | `chunk_id`、`document_id`、`text`、`embedding`、`metadata` |
| `RetrievedChunk` | 召回和重排阶段的统一结果 | `chunk_id`、`document_id`、`text`、`score`、`rank`、`recall_method`、`rerank_method`、`score_details` |
| `ConversationTurn` | 一轮完整对话 | `user_content`、`assistant_content` |
| `ChatHistory` | 多轮历史状态 | `summary`、`recent_turns` |

数据链路是：

```text
Document -> Chunk -> EmbeddedChunk -> RetrievedChunk
```

对话链路单独使用：

```text
ConversationTurn -> ChatHistory -> ChatMemory
```

## Reader

位置：`app/reader/`

Reader 负责把原始文件解析成统一的 `Document`。它只做“读取与解析”，不负责切分、向量化或入库。

### 支持格式

| 文件类型 | Reader | 说明 |
| --- | --- | --- |
| `.txt` | `TextReader` | 读取纯文本，并标记为 Markdown 文本 |
| `.md` | `MarkdownReader` | 读取 Markdown 文件 |
| `.pdf` | `MinerUPdfReader` | 调用 MinerU API 解析 PDF 为 Markdown |
| `.png` / `.jpg` / `.jpeg` | `MinerUImageReader` | 调用 MinerU API 解析图片为 Markdown |

统一入口：

```python
from app.reader import load_document

document = load_document("example.pdf")
```

### Text / Markdown Reader

`TextReader` 和 `MarkdownReader` 都会尝试多种编码读取文件：

```text
utf-8 -> utf-8-sig -> latin-1
```

返回的 `Document.metadata` 会包含：

- `reader`
- `source_format`
- `text_format`
- `char_count`

### MinerU Reader

PDF 和图片解析通过 `MinerUClient` 完成，依赖：

```text
MINERU_API_TOKEN
```

处理流程：

1. 申请上传 URL。
2. 上传本地文件。
3. 轮询解析结果。
4. 下载结果 ZIP。
5. 提取 `full.md`。
6. 构造成 `Document`。

PDF 默认开启 OCR、公式识别和表格识别；图片默认开启 OCR 和表格识别，不开启公式识别。

### 扩展 Reader

新增文件类型时：

1. 实现 `BaseReader.read()`。
2. 返回统一 `Document`。
3. 在 `reader_factory.py` 按后缀注册。
4. 确保 `Document.metadata["text_format"]` 能被 chunker 识别。

## Chunker

位置：`app/chunker/`

Chunker 负责把 `Document.text` 切成适合检索的 `Chunk`。当前只支持 Markdown 文本。

统一入口：

```python
from app.chunker import chunk_document

chunks = chunk_document(document)
```

### MarkdownChunker

默认参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `chunk_size` | `800` | 单个 chunk 的最大字符数 |
| `chunk_overlap` | `100` | 相邻 chunk 的强制重叠字符数 |
| `min_tail_chunk_ratio` | `0.25` | 尾部短 chunk 的最小长度比例 |
| `headers_to_split_on` | `#` / `##` / `###` | Markdown 标题层级 |

切分流程：

1. 使用 `MarkdownHeaderTextSplitter` 按标题结构切出 sections。
2. 对超过 `chunk_size` 的 section 使用 `RecursiveCharacterTextSplitter` 二次切分。
3. 对过短的尾部 chunk 合并到前一个 chunk。
4. 对同一 section 的相邻 chunk 补充固定 overlap。
5. 输出项目内定义的 `Chunk`。

每个 chunk 的 metadata 会包含：

- `source_path`
- `file_name`
- `file_type`
- `reader`
- `source_format`
- `text_format`
- `document_char_count`
- `chunk_index`
- `chunk_size`
- `headers`
- `header_path`

### 注意点

- 空文档会返回空列表。
- `chunk_overlap` 必须小于 `chunk_size`。
- `min_tail_chunk_ratio` 必须大于 `chunk_overlap / chunk_size`，否则尾部合并逻辑没有意义。

## Embedder

位置：`app/embedder/`

Embedder 负责把文本或查询转为向量。它不关心向量库，也不负责检索。

统一入口：

```python
from app.embedder import embed_chunks, get_embedder

embedded_chunks = embed_chunks(chunks, provider="bge")
embedder = get_embedder(provider="bge")
query_vector = embedder.embed_query("什么是遥感？")
```

### 支持 Provider

| provider | 类 | 默认模型 |
| --- | --- | --- |
| `bge` | `BGEEmbedder` | `BAAI/bge-small-zh-v1.5` |
| `minilm` | `SentenceTransformerEmbedder` | `all-MiniLM-L6-v2` |

### BGEEmbedder

关键参数：

- `model_name`
- `normalize_embeddings`
- `use_instruction`

当 `use_instruction=True` 时：

- 文档文本会加 `passage:`
- 查询文本会加 `query:`

`embed_chunks()` 会把 embedding 相关信息写入 metadata：

- `embedding_provider`
- `embedding_model`
- `normalize_embeddings`
- `use_instruction`

### 注意点

- Milvus collection 的向量维度必须和模型输出维度一致。
- 默认 pipeline 使用 `BAAI/bge-small-zh-v1.5`，维度为 512。
- 首次加载 sentence-transformers 模型会有明显耗时。

## Vector Store

位置：`app/vector_store/`

Vector Store 负责向量库读写。当前实现为 `MilvusVectorStore`。

统一入口：

```python
from app.vector_store import get_vector_store, save_embedded_chunks

vector_store = get_vector_store(provider="milvus", dimension=512)
save_embedded_chunks(embedded_chunks, provider="milvus")
```

### MilvusVectorStore

默认配置：

| 配置 | 默认值 | 环境变量 |
| --- | --- | --- |
| URI | `http://localhost:19530` | `MILVUS_URI` |
| database | `default` | `MILVUS_DB_NAME` |
| collection | `rag_chunks` | `MILVUS_COLLECTION_NAME` |
| token | 空 | `MILVUS_TOKEN` |
| mode | `local` | `MILVUS_MODE` |

Collection schema：

- `id`：Milvus 自增主键
- `chunk_id`：业务 chunk ID
- `document_id`
- `text`
- `embedding`
- `metadata`

索引配置：

- `index_type=HNSW`
- `metric_type=COSINE`
- `M=8`
- `efConstruction=64`

检索配置：

- `anns_field=embedding`
- `metric_type=COSINE`
- `ef=64`

### 重要方法

- `create_collection()`：collection 不存在时创建 collection 和索引
- `insert()`：写入 `EmbeddedChunk`
- `search()`：根据 query vector 返回 `RetrievedChunk`
- `get_chunk_text_by_id()`：按业务 `chunk_id` 查询原文
- `drop_collection()`：删除当前 collection

### 注意点

`drop_collection()` 和入库脚本会删除当前 collection，执行前必须确认 `.env` 指向的库是目标库。

## Recaller

位置：`app/recaller/`

Recaller 负责第一阶段召回。当前实现是纯向量召回。

```python
from app.recaller import get_recaller

recaller = get_recaller(provider="vector")
results = recaller.recall("什么是遥感？", top_k=30, score_threshold=0.4)
```

### VectorRecaller

处理流程：

1. 校验 query、`top_k` 和 `score_threshold`。
2. 调用 `embedder.embed_query()` 得到 query vector。
3. 调用 `vector_store.search()`。
4. 按召回分数阈值过滤。
5. 给结果补充：
   - `rank`
   - `recall_method="vector"`
   - `score_details["vector_score"]`

默认参数：

- `top_k=30`
- `score_threshold=0.4`

## Reranker

位置：`app/reranker/`

Reranker 负责第二阶段精排。当前实现是 BGE cross-encoder。

```python
from app.reranker import get_reranker

reranker = get_reranker(provider="bge")
reranked = reranker.rerank(query, candidates, top_n=10, score_threshold=0.5)
```

### BGEReranker

默认模型在 `RAGPipeline` 中配置为：

```text
BAAI/bge-reranker-base
```

`BGEReranker` 自身默认值为：

```text
BAAI/bge-reranker-v2-m3
```

如果从 pipeline 使用，以 pipeline 的参数为准。

处理流程：

1. 把 query 和每个 candidate text 组成 pair。
2. 调用 CrossEncoder 预测相关性分数。
3. 将分数写入：
   - `candidate.score`
   - `candidate.score_details["rerank_score"]`
   - `candidate.rerank_method="bge"`
4. 按阈值过滤。
5. 按分数降序排序。
6. 返回前 `top_n`。

环境变量：

- `RERANKER_DEVICE`：如 `cuda` 或 `cpu`
- `RERANK_BATCH_SIZE`：推理 batch size

如果没有显式 batch size，代码会根据 CUDA 可用性和显存做保守估计。

## Prompt Builder

位置：`app/prompt_builder/`

Prompt Builder 负责把用户问题、检索结果和历史对话组织成模型输入。

支持 provider：

| provider | 类 | 输出 |
| --- | --- | --- |
| `chat` / `chat_messages` | `ChatMessagesPromptBuilder` | `list[dict[str, str]]` |
| `string` | `StringPromptBuilder` | `str` |

### StringPromptBuilder

适合调试或 completion-style 模型。它会输出：

```text
system prompt

参考资料：
...

用户问题：
...
```

关键参数：

- `system_prompt`
- `max_context_chars`
- `allow_general_fallback`

当 `allow_general_fallback=True` 时，如果参考资料不足，提示词允许模型明确说明资料不足后使用通用知识补充；否则要求严格只依据参考资料回答。

### ChatMessagesPromptBuilder

默认用于 OpenAI Chat / Responses API。输出结构：

```python
[
    {"role": "system", "content": "..."},
    *history_messages,
    {"role": "user", "content": "..."}
]
```

它会过滤历史中的 system message，避免多个 system prompt 混在一起。

## Generator

位置：`app/generator/`

Generator 负责调用 LLM 生成答案，不负责检索、prompt 构建或记忆管理。

统一入口：

```python
from app.generator import generate, stream_generate

answer = generate(prompt)
for delta in stream_generate(prompt):
    print(delta, end="")
```

### OpenAIGenerator

当前唯一实现为 `OpenAIGenerator`，使用 OpenAI Responses API：

```python
client.responses.create(
    model=model,
    input=response_input,
    stream=True,
    **request_kwargs,
)
```

默认模型：

```text
gpt-5.4-mini
```

配置：

- `OPENAI_API_KEY`
- `base_url` 可通过构造参数传入，用于 OpenAI 兼容接口
- `request_kwargs` 可传递额外 Responses API 参数

输入支持：

- 字符串 prompt
- chat messages：`list[dict[str, str]]`

## Memory

位置：`app/memory/`

`ChatMemory` 管理多轮对话历史。它只保存真实发生过的用户问题和助手最终回答。

不会保存：

- system prompt
- RAG 参考资料
- 当前轮未完成的 assistant 输出

默认窗口：

```text
max_turns=5
```

当历史超过窗口时，会把较早的一轮对话和已有 summary 合并成新的 summary。该压缩过程会调用 generator，因此需要可用的 LLM 配置。

主要方法：

- `add_turn()`
- `get_messages()`
- `clear()`
- `has_history()`
- `get_history()`

## Query Rewriter

位置：`app/query_rewriter/`

`LLMQueryRewriter` 用于多轮 RAG 中的检索问题改写。

行为：

- 没有历史时，直接返回当前 query。
- 有历史时，调用 generator 把当前问题改写为独立、明确、适合检索的问题。
- 如果模型返回空字符串，回退到原 query。

它只负责改写检索 query，不回答问题，也不修改 memory。

## Pipeline

位置：`app/pipeline/`

`RAGPipeline` 是在线问答总编排。

默认配置摘要：

| 参数 | 默认值 |
| --- | --- |
| `embedder_provider` | `bge` |
| `embedder_model_name` | `BAAI/bge-small-zh-v1.5` |
| `vector_store_provider` | `milvus` |
| `dimension` | `512` |
| `recaller_provider` | `vector` |
| `reranker_provider` | `bge` |
| `reranker_model_name` | `BAAI/bge-reranker-base` |
| `prompt_builder_provider` | `chat` |
| `prompt_max_context_chars` | `12000` |
| `generator_provider` | `openai` |
| `generator_model` | `gpt-5.4-mini` |
| `memory_max_turns` | `5` |
| `enable_query_rewriter` | `True` |

主要方法：

- `rewrite_query()`：根据历史改写检索问题
- `retrieve()`：执行召回和重排
- `build_prompt()`：构造当前轮 prompt
- `ask()`：执行完整 RAG 问答
- `clear_memory()`：清空内部记忆

`RAGPipeline` 支持传入已有组件实例，如 `embedder`、`vector_store`、`recaller`、`reranker`、`prompt_builder`、`generator`、`memory` 和 `query_rewriter`。这让测试和模块替换更容易。

## Backend

位置：`backend/`

后端使用 FastAPI，入口为 `backend/server.py`。

启动时：

1. 读取 `.env`。
2. 清空会话缓存。
3. 初始化全局 `RAGPipeline(enable_query_rewriter=True)`。

核心接口：

- `GET /api/health`
- `POST /api/chat/stream`
- `DELETE /api/chat/session/{session_id}`

`/api/chat/stream` 返回 Server-Sent Events，事件包括：

- `session`
- `delta`
- `done`
- `error`

## Frontend

位置：`frontend/`

前端使用 Vue 3 + Vite。主要文件：

- `frontend/src/App.vue`
- `frontend/src/style.css`

主要功能：

- 调用后端 `/api/chat/stream`
- 解析 SSE block
- 处理 `session`、`delta`、`error` 事件
- 本地维护 `sessionId`
- 支持停止流式生成
- 支持新建对话并清理后端 session

配置：

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```
