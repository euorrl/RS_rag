# 快速开始

## 环境要求

- Python 3.10
- Node.js 和 npm
- Docker / Docker Compose
- OpenAI 兼容 API Key
- 如需解析 PDF 或图片，需要 MinerU API Token

## 安装后端依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 启动本地 Milvus

```bash
docker compose up -d
```

服务端口：

- Milvus：`19530`
- MinIO：`9000` / `9001`
- Attu：`8000`

## 配置环境变量

在项目根目录创建 `.env`：

```text
OPENAI_API_KEY=...
MINERU_API_TOKEN=...

MILVUS_MODE=local
MILVUS_URI=http://localhost:19530
MILVUS_DB_NAME=default
MILVUS_COLLECTION_NAME=rag_chunks

FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
CHAT_MEMORY_MAX_TURNS=5
RERANKER_DEVICE=
RERANK_BATCH_SIZE=
```

使用 Zilliz Cloud 时，可改为：

```text
MILVUS_MODE=cloud
MILVUS_URI=...
MILVUS_TOKEN=...
MILVUS_DB_NAME=default
MILVUS_COLLECTION_NAME=rag_chunks
```

## 构建资料库

默认入库脚本读取：

```text
data/remote_sensing_fundamentals/*.pdf
```

执行：

```bash
python scripts/create_cloud_milvus_database.py
```

该脚本会 **删除并重建** 当前配置指向的 collection，然后解析 PDF、切分 Markdown、生成 embedding 并写入 Milvus。

## 启动服务

后端：

```bash
python scripts/run_backend.py
```

前端：

```bash
cd frontend
npm install
npm run dev
```

如果后端不是默认地址，可创建 `frontend/.env.local`：

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 命令行问答

```bash
python scripts/run_rag_chat.py
```
