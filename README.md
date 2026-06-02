# RS RAG

RS RAG 是一个面向遥感学习资料的 RAG 问答项目，包含文档解析、切分、向量化、Milvus 检索、BGE 重排、OpenAI 兼容模型生成，以及 FastAPI + Vue 的聊天界面。

详细架构、模块说明、API、评估和部署文档请看 `docs/`，或通过 MkDocs / Read the Docs 构建后的文档站阅读。

快速访问: https://rs-rag.vercel.app/

项目详细文档: https://rs-rag.readthedocs.io/zh-cn/latest/

数据库资料来源: https://zh.z-lib.fm/book/dAKJkEV7g7/%E9%81%A5%E6%84%9F%E5%AF%BC%E8%AE%BA.html

> 注意: 本项目的后端模型服务不会长期在线(抱歉🙃)，如果无法访问或需要体验，欢迎联系Email-a1913397362@163.com

## 本地启动

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
docker compose up -d
```

在项目根目录创建 `.env`：

```text
OPENAI_API_KEY=...
MINERU_API_TOKEN=...

MILVUS_MODE=local
MILVUS_URI=http://localhost:19530
MILVUS_DB_NAME=default
MILVUS_COLLECTION_NAME=rag_chunks

FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

检查 Milvus：

```bash
python scripts/check_milvus_connection.py
```

构建资料库：

```bash
python scripts/create_cloud_milvus_database.py
```

终端体验：
```bash
python scripts/run_rag_chat.py
```

启动后端：

```bash
python scripts/run_backend.py
```

启动前端：

```bash
cd frontend
npm install
npm run dev
```

## 常用命令

```bash
python scripts/run_rag_chat.py
python scripts/run_evaluation.py
pytest
```

前端构建：

```bash
cd frontend
npm run build
```

文档本地预览：

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## 项目结构

```text
app/          RAG 核心模块
backend/      FastAPI 后端
frontend/     Vue 3 + Vite 前端
evaluation/   检索与生成评估
scripts/      启动、入库、迁移和评估脚本
tests/        测试
docs/         项目文档
```

## 文档入口

- [项目概览](docs/index.md)
- [快速开始](docs/getting-started.md)
- [系统架构](docs/architecture.md)
- [模块说明](docs/modules.md)
- [后端 API](docs/api.md)
- [评估](docs/evaluation.md)
- [开发与部署](docs/development.md)
