# RS RAG 文档

RS RAG 是一个面向遥感学习资料的 RAG 问答项目。系统支持把文本、Markdown、PDF 和图片资料解析为统一文档结构，写入 Milvus 向量库，并通过 FastAPI 后端和 Vue 前端提供流式多轮问答。

## 项目能力

- 文档解析：支持 `.txt`、`.md`、`.pdf`、`.png`、`.jpg`、`.jpeg`
- 文本切分：基于 Markdown 标题结构和递归字符切分
- 向量化：默认使用 BGE embedding，也支持 MiniLM
- 向量库：默认使用 Milvus / Zilliz Cloud
- 检索链路：向量召回 + BGE cross-encoder 重排
- 生成链路：OpenAI 兼容 Responses API 流式生成
- 多轮对话：会话级 `ChatMemory` 和 LLM query rewrite
- Web 体验：FastAPI SSE 接口 + Vue 3 聊天界面
- 评估：检索召回评估与 LLM-as-judge 生成评估

## 推荐阅读顺序

1. [快速开始](getting-started.md)
2. [系统架构](architecture.md)
3. [模块说明](modules.md)
4. [后端 API](api.md)
5. [评估](evaluation.md)
6. [开发与部署](development.md)

## 代码入口

- `app/pipeline/rag_pipeline.py`：RAG 问答总编排
- `backend/server.py`：FastAPI 应用和 SSE 接口
- `frontend/src/App.vue`：前端聊天页面
- `scripts/create_cloud_milvus_database.py`：资料入库脚本
- `scripts/run_backend.py`：后端启动脚本
- `scripts/run_evaluation.py`：评估入口
