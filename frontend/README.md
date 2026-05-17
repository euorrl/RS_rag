# RS RAG Frontend

Vue 3 + Vite 前端，用于连接 `backend` 中的 RAG 流式问答接口。

## 本地启动

```bash
npm install
npm run dev
```

默认后端地址是：

```text
http://127.0.0.1:8000
```

如果后端部署在其他地址，可以创建 `frontend/.env.local`：

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 构建

```bash
npm run build
```
