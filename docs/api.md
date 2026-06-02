# 后端 API

后端使用 FastAPI，入口为 `backend/server.py`。

## 启动

```bash
python scripts/run_backend.py
```

默认监听：`http://127.0.0.1:8000`

## 健康检查

```http
GET /api/health
```

返回示例：

```json
{
  "status": "ok",
  "pipeline_ready": true,
  "session_count": 1
}
```

## 流式问答

```http
POST /api/chat/stream
Content-Type: application/json
```

请求体：

```json
{
  "question": "什么是遥感？",
  "session_id": "optional-session-id"
}
```

约束：

- `question` 必填，最大长度 500
- `session_id` 可选，最大长度 128
- 空白问题会返回 `400`
- pipeline 未初始化时会返回 `503`

返回类型：`text/event-stream`

事件类型：

| event | data | 说明 |
| --- | --- | --- |
| `session` | `{"session_id": "..."}` | 返回新建或复用的会话 ID |
| `delta` | 文本片段 | 模型流式生成的增量文本 |
| `done` | `{}` | 本轮生成完成 |
| `error` | 错误文本 | 本轮处理异常 |

## 清除会话

```http
DELETE /api/chat/session/{session_id}
```

返回示例：

```json
{
  "session_id": "session-1",
  "cleared": true
}
```

该接口只删除后端内存中的 `ChatMemory`，不会删除向量库数据。

## CORS

允许来源由 `FRONTEND_ORIGINS` 配置，默认：

```text
http://localhost:5173,http://127.0.0.1:5173
```
