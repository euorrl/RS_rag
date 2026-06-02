# 开发与部署

## 测试

运行全部测试：

```bash
pytest
```

按模块运行：

```bash
pytest tests/reader
pytest tests/chunker
pytest tests/pipeline
pytest tests/backend
```

部分集成测试需要真实模型、外部 API 或 Milvus。

## 前端开发

```bash
cd frontend
npm install
npm run dev
```

构建：

```bash
cd frontend
npm run build
```

前端通过 `VITE_API_BASE_URL` 配置后端地址。

## 文档开发

文档位于 `docs/`，使用 MkDocs Material。

```bash
pip install -r docs/requirements.txt
mkdocs serve
mkdocs build --strict
```

## Read the Docs

Read the Docs 配置文件为 `.readthedocs.yaml`。

当前策略：

- 使用 Ubuntu 22.04
- 使用 Python 3.10
- 使用 `mkdocs.yml` 作为 MkDocs 配置
- 只安装 `docs/requirements.txt`

这样可以避免文档构建时安装 torch、transformers、sentence-transformers 等较重的运行时依赖。

## 扩展指南

新增 Reader：实现 `BaseReader`，在 `app/reader/reader_factory.py` 注册，并返回统一 `Document`。

新增 Chunker：实现 `BaseChunker`，在 `app/chunker/chunker_factory.py` 注册，并返回 `list[Chunk]`。

新增 Embedder：实现 `BaseEmbedder` 的 `embed_texts()` 和 `embed_query()`，在 `app/embedder/embedder_factory.py` 注册。

新增 Vector Store：实现 `BaseVectorStore`，在 `app/vector_store/vector_store_factory.py` 注册，并返回统一 `RetrievedChunk`。

新增 Generator：实现 `BaseGenerator` 的 `generate()` 和 `stream_generate()`，在 `app/generator/generator_factory.py` 注册。

## 注意事项

- `scripts/create_cloud_milvus_database.py` 会清空并重建当前 collection。
- `scripts/copy_local_milvus_to_cloud.py` 会重建目标 collection。
- BGE embedding 和 reranker 首次运行会加载模型，耗时和显存取决于环境。
- MinerU 解析、LLM功能需要有效的API_TOKEN`。
