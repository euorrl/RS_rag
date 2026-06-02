# 评估

项目提供检索评估和生成评估两类能力。

## 数据集

默认数据集：

```text
evaluation/dataset/eval_dataset.json
```

样本结构：

```json
{
  "id": "q001",
  "question": "问题文本",
  "chunks": [
    ["golden-chunk-id-1"],
    ["golden-chunk-id-2a", "golden-chunk-id-2b"]
  ]
}
```

`chunks` 是 golden evidence 分组。一个内层列表表示同一个 evidence 的可替代 chunk，只要召回其中任意一个就算覆盖该 evidence。

## 运行评估

```bash
python scripts/run_evaluation.py
```

结果输出：

```text
evaluation/results/evaluation_result.json
```

## 检索评估

入口：`evaluation/pipeline/retrieval_pipeline.py`

默认指标：

- `evidence_recall@10`
- `mrr@10`
- `mean_retrieval_seconds`
- `mean_recall_seconds`
- `mean_rerank_seconds`

## 生成评估

入口：`evaluation/pipeline/generation_pipeline.py`

默认流程：

1. 使用默认 recaller 和 reranker 获取 context。
2. 构造 strict RAG prompt。
3. 调用 generator 生成答案。
4. 用 LLM-as-judge 评估忠实度和相关性。

### Claim Faithfulness

流程：

1. 从答案中抽取 atomic factual claims。
2. 判断每个 claim 是否被 retrieved context 支持。
3. 将 `SUPPORTED`、`PARTIALLY_SUPPORTED`、`UNSUPPORTED` 映射为 1.0、0.5、0.0。

### Answer Relevance

流程：

1. 仅根据答案反推 1-3 个可能问题。
2. 判断反推问题是否与原始问题语义匹配。
3. 输出 `RELEVANT`、`PARTIALLY_RELEVANT` 或 `IRRELEVANT`。
