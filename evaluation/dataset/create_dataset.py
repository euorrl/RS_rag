import json
import sys
from pathlib import Path

from app.generator import generate
from app.vector_store.milvus_store import MilvusVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHUNK_IDS = [
    "3e5d698b-1862-4970-add5-8388b3f57fa9",
]

DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"


def main() -> None:
    """根据一组 chunk_id 生成一个问题，并追加写入 eval_dataset.json。"""

    vector_store = MilvusVectorStore()

    texts = []
    for chunk_id in CHUNK_IDS:
        text = vector_store.get_chunk_text_by_id(chunk_id)
        texts.append(f"chunk_id: {chunk_id}\n{text}")

    context = "\n\n".join(texts)

    prompt = f"""
请根据下面这些 chunk 内容，生成一个适合作为 RAG 评估集的问题。

要求：
1. 问题必须能由这些 chunk 回答。
2. 尽量让问题需要用到这些 chunk 的核心信息。
3. 不要提到“chunk”“资料”“上下文”这些词。
4. 只返回问题本身，不要解释。

内容如下：

{context}
""".strip()

    question = generate(prompt, model="gpt-5.5").strip()

    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    # 先把已有数据重新编号，避免手动删除后 id 不连续或重复
    for i, sample in enumerate(data, start=1):
        sample["id"] = f"q{i:03d}"

    item = {
        "id": f"q{len(data) + 1:03d}",
        "question": question,
        "chunks": [[chunk_id] for chunk_id in CHUNK_IDS],
    }

    data.append(item)

    DATASET_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
