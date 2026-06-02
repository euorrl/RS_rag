# 评估

项目提供两类评估：

- 检索评估：评估系统是否能把应命中的资料 chunk 召回到前排。
- 生成评估：评估最终答案是否忠实于检索上下文，以及是否回答了原问题。

默认入口：

```bash
python scripts/run_evaluation.py
```

该脚本会依次运行检索评估和生成评估，并把结果写入：

```text
evaluation/results/evaluation_result.json
```

## 数据集

默认数据集：

```text
evaluation/dataset/eval_dataset.json
```

每条样本结构如下：

```json
{
  "id": "q001",
  "question": "问题文本",
  "chunks": [
    ["golden-chunk-id-1"],
    ["golden-chunk-id-2"],
    ...
  ]
}
```

字段含义：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | `str` | 问题编号，脚本会使用 `q001`、`q002` 这类格式 |
| `question` | `str` | 用于评估的用户问题 |
| `chunks` | `list[str]` | golden evidence chunks |


## 数据集构建

数据集构建脚本：

```text
evaluation/dataset/create_dataset.py
```

当前脚本采用“先指定证据 chunk，再生成问题”的方式：

1. 在脚本中的 `CHUNK_IDS` 填入一个或多个目标 chunk ID。
2. 脚本用 `MilvusVectorStore.get_chunk_text_by_id()` 从向量库读取这些 chunk 原文。
3. 将 chunk 内容拼接成 context。
4. 调用 LLM 生成一个适合作为评估集的问题。
5. 读取已有 `eval_dataset.json`。
6. 重新整理已有样本编号，避免 ID 不连续或重复。
7. 追加新样本，`chunks` 字段记录本次使用的 golden chunk IDs。

运行：

```bash
python evaluation/dataset/create_dataset.py
```

新增样本会写回：

```text
evaluation/dataset/eval_dataset.json
```

### 构建样本时的原则

推荐让一个问题对应明确的资料证据：

- 问题必须能由指定 chunk 回答。
- 尽量覆盖 chunk 中的核心信息，而不是只问边缘细节。
- 如果一个问题必须依赖多个证据点，应该把多个 evidence 分别写入 `chunks` 的不同内层列表。

### 数据集构建的前置条件

构建数据集前需要确保：

- Milvus / Zilliz Cloud 中已经写入了对应 chunks。
- `.env` 中的 Milvus 配置指向目标 collection。
- `CHUNK_IDS` 中的 chunk ID 能通过 `get_chunk_text_by_id()` 查到原文。
- LLM 服务可用，因为问题生成需要调用 `generate()`。

## 评估运行流程

`scripts/run_evaluation.py` 会读取：

```text
evaluation/dataset/eval_dataset.json
```

然后运行：

```python
run_retrieval_pipeline(dataset_path=DATASET_PATH)
run_generation_pipeline(dataset_path=DATASET_PATH)
```

最终结果结构大致为：

```json
{
  "dataset_path": "evaluation/dataset/eval_dataset.json",
  "retrieval": {
    "total": 50,
    "mean_evidence_recall@10": 0.0,
    "mean_mrr@10": 0.0
  },
  "generation": {
    "total": 50,
    "mean_claim_faithfulness": 0.0,
    "mean_answer_relevance": 0.0,
    "items": []
  }
}
```

实际文件中还会包含逐样本指标、检索耗时、召回耗时、重排耗时等字段。

## 检索评估

入口：

```text
evaluation/pipeline/retrieval_pipeline.py
```

默认参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DEFAULT_K` | `10` | 指标只看前 10 个结果 |
| `DEFAULT_RECALL_TOP_K` | `30` | 向量召回最多返回 30 个候选 |
| `DEFAULT_RECALL_SCORE_THRESHOLD` | `0.4` | 召回阶段分数阈值 |
| `DEFAULT_RERANK_TOP_N` | `10` | 重排后最多返回 10 个结果 |
| `DEFAULT_RERANK_SCORE_THRESHOLD` | `0.5` | 重排阶段分数阈值 |

检索 pipeline 会：

1. 初始化默认 recaller。
2. 初始化默认 reranker。
3. 对每个问题执行向量召回和 BGE 重排。
4. 提取返回结果中的 `chunk_id`。
5. 与样本中的 golden chunks 计算指标。

### 指标 1：Evidence Recall@K

函数：

```text
evaluation/retrieval/metrics.py::evidence_recall_at_k
```

含义：

```text
前 K 个检索结果覆盖了多少 golden evidence 分组。
```

计算方式：

```text
Evidence Recall@K = 被覆盖的 evidence 组数 / 总 evidence 组数
```

例子：

```text
retrieved_ids 前 10 = [A, B, C]
golden_chunks = [A, B, E]
```

覆盖情况：

- `[A]` 被覆盖。
- `[B]` 被覆盖。
- `[E]` 未覆盖。

所以：

```text
Evidence Recall@10 = 2 / 3 = 0.6667
```

这个指标关注“证据有没有被找回来”，不关心第几个位置命中。

### 指标 2：MRR@K

函数：

```text
evaluation/retrieval/metrics.py::mrr_at_k
```

MRR 全称是 Mean Reciprocal Rank，平均倒数排名。项目里先计算每个问题的 `MRR@K`，再对所有样本取平均。

单条问题的计算方式：

```text
MRR@K = 1 / 第一个命中的 golden chunk 排名
```

如果前 K 个结果没有命中任何 golden chunk，则为 `0.0`。

例子：

```text
retrieved_ids 前 10 = [X, Y, A, C]
golden_chunks = [A, B]
```

第一个命中的是 `A`，排名第 3：

```text
MRR@10 = 1 / 3 = 0.3333
```

这个指标关注“第一个正确证据排得有多靠前”。越早命中，分数越高。

## 生成评估

入口：

```text
evaluation/pipeline/generation_pipeline.py
```

默认流程：

1. 使用默认 recaller 和 reranker 获取 retrieved chunks。
2. 将 retrieved chunks 拼接成 context。
3. 构造 strict RAG prompt。
4. 调用 generator 生成答案。
5. 用 LLM-as-judge 评估答案质量。

生成评估使用两个核心指标：

- `claim_faithfulness`
- `answer_relevance`

### 指标 3：Claim Faithfulness

相关文件：

```text
evaluation/generation/claim_extractor.py
evaluation/generation/faithfulness_judge.py
evaluation/generation/metrics.py
```

该指标评估：

```text
答案中的事实断言是否被 retrieved context 支持。
```

流程：

1. `extract_claims(answer)` 从答案中抽取 atomic factual claims。
2. `judge_claim_support(context, claims)` 判断每个 claim 是否被 context 支持。
3. `compute_claim_faithfulness(labels)` 把标签映射为分数并取平均。

标签映射：

| 标签 | 分数 | 含义 |
| --- | --- | --- |
| `SUPPORTED` | `1.0` | context 明确支持该 claim |
| `PARTIALLY_SUPPORTED` | `0.5` | context 部分支持，但 claim 有扩展或需要轻微推断 |
| `UNSUPPORTED` | `0.0` | context 不支持，或与 context 矛盾 |

公式：

```text
Claim Faithfulness = 所有 claim 支持分数的平均值
```

如果没有抽取到 claims，项目当前返回：

```text
0.0
```

注意：这个指标评估的是“是否忠实于检索上下文”，不是判断答案在真实世界中是否正确。即使某个事实真实存在，只要当前 context 没有支持，也应判为不支持。

### 指标 4：Answer Relevance

相关文件：

```text
evaluation/generation/question_generator.py
evaluation/generation/answer_relevance_judge.py
evaluation/generation/metrics.py
```

该指标评估：

```text
答案是否真正回答了原始问题。
```

流程：

1. `generate_questions_from_answer(answer)` 仅根据答案反推出 1-3 个可能问题。
2. `judge_answer_relevance(question, generated_questions)` 判断反推问题是否与原始问题语义匹配。
3. `normalize_answer_relevance_score(score, label)` 将标签或分数规范化为 `0.0`、`0.5` 或 `1.0`。

标签映射：

| 标签 | 分数 | 含义 |
| --- | --- | --- |
| `RELEVANT` | `1.0` | 答案直接回答了原问题 |
| `PARTIALLY_RELEVANT` | `0.5` | 答案只回答了一部分，或有所偏移 |
| `IRRELEVANT` | `0.0` | 答案基本没有回答原问题 |

这个指标不判断答案事实是否正确，也不判断是否来自 context，只看答案和问题是否语义对齐。

## 四个核心指标的分工

| 指标 | 阶段 | 关注点 | 越高表示 |
| --- | --- | --- | --- |
| `evidence_recall@10` | 检索 | golden evidence 是否被召回 | 需要的证据覆盖更完整 |
| `mrr@10` | 检索 | 第一个正确证据排得多靠前 | 正确证据更早出现 |
| `claim_faithfulness` | 生成 | 答案事实是否被 context 支持 | 幻觉更少，更忠实 |
| `answer_relevance` | 生成 | 答案是否回答原问题 | 回答更贴题 |

## 当前评估结果与解读

当前 `evaluation/results/evaluation_result.json` 中记录了一次完整评估结果，评估集规模为 50 条样本。

主要结果如下：

| 指标 | 当前结果 | 说明 |
| --- | ---: | --- |
| `mean_evidence_recall@10` | `0.847` | 平均每个问题约 84.7% 的 golden evidence 能在前 10 个结果中被覆盖 |
| `mean_mrr@10` | `0.970` | 第一个正确证据通常排在非常靠前的位置 |
| `mean_claim_faithfulness` | `0.9591` | 生成答案中的事实断言大多能被 retrieved context 支持 |
| `mean_answer_relevance` | `1.000` | 生成答案基本都能回答评估问题 |

检索耗时结果如下：

| 指标 | 当前结果 |
| --- | ---: |
| `mean_retrieval_seconds` | `2.528s` |
| `mean_recall_seconds` | `1.427s` |
| `mean_rerank_seconds` | `1.101s` |
| `pipeline_init_seconds` | `117.540s` |

其中 `pipeline_init_seconds` 主要包含 embedder、Milvus 连接、reranker 等组件初始化与模型加载时间；单条问题的在线检索耗时应该参考 `mean_retrieval_seconds`、`mean_recall_seconds` 和 `mean_rerank_seconds`，不同设备环境的运行时间会有出入。

### 结果偏高的原因

这组评估结果整体较高，尤其是 `mean_mrr@10 = 0.970` 和 `mean_answer_relevance = 1.000`。这说明当前系统在该评估集上表现很好，但也需要注意：**当前评估集的问题选取较为规范，因此指标会明显偏高。**

当前评估集的构建方式是先指定 golden chunks，再基于这些 chunk 生成或整理问题。这种方式有利于构造可控评估集，但也会带来几个特点：

- 问题通常与资料表述高度相关，关键词和语义都比较清晰。
- 问题边界明确，较少出现真实用户常见的省略、口语化、错别字或模糊指代。
- 问题通常能由指定 chunk 直接回答，噪声样本和不可回答样本较少。
- 多数问题不包含复杂多轮上下文依赖，因此对 query rewrite 的压力较小。
- 评估问题覆盖的是资料中的标准知识点，和真实开放式提问还有差距。

因此，这组结果更适合理解为：

```text
系统在规范、资料内、证据明确的问题集上的表现。
```

它不完全等价于真实用户场景下的表现。真实用户可能会提出更短、更模糊、更口语化、跨章节或资料外的问题，这些情况可能导致召回、重排和生成指标下降。

### 后续评估建议

为了更接近真实使用场景，可以继续补充几类测试样本：

- 口语化问题：例如“这个有啥用”“它和普通拍照有啥区别”
- 指代型问题：依赖上一轮上下文的问题，用于测试 query rewrite
- 资料外问题：用于测试系统是否能说明资料不足
- 干扰性问题：关键词相似但答案属于不同章节
- 多证据问题：需要跨多个 chunk 才能完整回答
- 人工编写问题：减少由 golden chunk 反向生成问题带来的偏置

这样可以让评估结果更稳健，也更接近实际用户体验。

## 使用注意

- 检索评估依赖 Milvus 中的 chunk IDs 和数据集里的 golden chunk IDs 一致。
- 生成评估会多次调用 LLM，运行时间和成本高于检索评估。
- `scripts/run_evaluation.py` 会覆盖写入 `evaluation/results/evaluation_result.json`。
